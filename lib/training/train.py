import os
import time
import functools
import dataclasses
from absl import logging

import flax
import jax
import numpy as np
import jax.numpy as jnp

from clu import metric_writers

from .metrics import cross_entropy_loss, soft_cross_entropy_loss, mean_squared_loss, get_metrics
from .utils import create_learning_rate_fn, create_train_state
from .utils import train_step, eval_step, process_batch
from .utils import save_checkpoint, restore_checkpoint


def sync_batch_stats(state):
    """
    Sync the batch statistics across devices.
    Args:
        state (train_state.TrainState): Training state.

    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def jit_fn(state, loss_type, l2_reg, use_pmap=False, has_feat=False, has_bn=False):
    if use_pmap:
        state = flax.jax_utils.replicate(state)
        jit_train_step = jax.pmap(
            functools.partial(train_step, loss_type=loss_type, l2_reg=l2_reg, has_feat=has_feat, has_bn=has_bn),
            axis_name='batch')
        jit_eval_step = jax.pmap(functools.partial(eval_step, loss_type=loss_type, has_feat=has_feat, has_bn=has_bn),
                                 axis_name='batch')
    else:
        jit_train_step = jax.jit(
            functools.partial(train_step, loss_type=loss_type, l2_reg=l2_reg, has_feat=has_feat, has_bn=has_bn))
        jit_eval_step = jax.jit(functools.partial(eval_step, loss_type=loss_type, has_feat=has_feat, has_bn=has_bn))

    return state, jit_train_step, jit_eval_step


def train_and_evaluate(config, model, dataset, workdir, use_pmap, dtype=jnp.float32, params=None, batch_stats=None,
                       mask=None, has_feat=False, has_bn=False, writer=None, transform=None, name='',
                       steps_per_log=None, steps_per_checkpoint=None, steps_per_eval=None):
    # --------------------------------------
    # Setup
    # --------------------------------------
    ds_train, ds_test = dataset

    if writer is None:
        writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    if transform is None:
        transform = lambda rng, x: x

    steps_per_epoch = config.train_size // config.batch_size
    if steps_per_checkpoint is None:
        steps_per_checkpoint = steps_per_epoch * 5

    if steps_per_eval is None:
        steps_per_eval = steps_per_epoch * 1

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    print(steps_per_epoch, config.train_size, config.batch_size, config.num_epochs)

    logging.info('Number of training steps: {}'.format(num_steps))

    rng = jax.random.PRNGKey(config.random_seed)
    rng, init_rng = jax.random.split(rng)

    # --------------------------------------
    # Initialize Model and Optimizer
    # --------------------------------------
    learning_rate_fn = create_learning_rate_fn(
        config.learning_rate, steps_per_epoch, config.num_epochs, config.warmup_epochs)
    state = create_train_state(init_rng, config, model, learning_rate_fn, params=params, batch_stats=batch_stats,
                               mask=mask, has_bn=has_bn)

    step, step_offset, best_val_acc, accuracy = 0, 0, 0.0, 0.0
    if config.resume:
        state = restore_checkpoint(state, workdir)
        step_offset = jax.device_get(state.step)
        best_val_acc = jax.device_get(state.best_val_acc)

    # --------------------------------------
    # Training
    # --------------------------------------
    train_metrics = []

    logging.info('Loss type: {}'.format(config.loss_type))
    if config.loss_type == 'xent':
        loss_type = cross_entropy_loss
    elif config.loss_type == 'soft_xent':
        loss_type = soft_cross_entropy_loss
    elif config.loss_type == 'mse':
        loss_type = mean_squared_loss
    else:
        raise ValueError('Unknown loss type: {}'.format(config.loss_type))
    state, jit_train_step, jit_eval_step = jit_fn(state, loss_type, config.l2_reg, use_pmap=use_pmap, has_feat=has_feat,
                                                  has_bn=has_bn)

    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(range(step_offset, num_steps), ds_train.as_numpy_iterator()):
        rng, train_rng = jax.random.split(rng)
        if use_pmap:
            train_rng = flax.jax_utils.replicate(train_rng)

        image, label = process_batch(batch, use_pmap=use_pmap, dtype=dtype)
        image = transform(train_rng, image)
        state, metrics = jit_train_step(state, {'image': image, 'label': label}, train_rng)

        if step == step_offset:
            logging.info('Initial compilation completed. Elapsed: {}s'.format(
                time.time() - train_metrics_last_t))
            train_metrics_last_t = time.time()

        if (step + 1) % steps_per_log == 0:
            train_metrics.append(metrics)
            if (step + 1) % steps_per_log == 0:
                train_metrics = get_metrics(train_metrics, use_pmap=use_pmap)
                summary = {
                    f'{name}/train_{k}': v
                    for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = steps_per_log / (
                        time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % (steps_per_eval) == 0:
            epoch = (step + 1) // steps_per_epoch
            eval_metrics = []

            for batch in ds_test.as_numpy_iterator():
                image, label = process_batch(batch, use_pmap=use_pmap, dtype=dtype)
                metrics = jit_eval_step(state, {'image': image, 'label': label})
                eval_metrics.append(metrics)

            eval_metrics = get_metrics(eval_metrics, use_pmap=use_pmap)
            summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
            writer.write_scalars(
                step + 1, {f'{name}/eval_{key}': val for key, val in summary.items()})
            writer.flush()
            accuracy = summary['accuracy'] * 100

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            if use_pmap:
                state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

            if accuracy > best_val_acc:
                best_val_acc = accuracy
                if use_pmap:
                    state = dataclasses.replace(
                        state, **{'step': flax.jax_utils.replicate(step), 'epoch': flax.jax_utils.replicate(epoch),
                                  'best_val_acc': flax.jax_utils.replicate(best_val_acc)})
                else:
                    state = dataclasses.replace(
                        state, **{'step': step, 'epoch': epoch, 'best_val_acc': best_val_acc})

                try:
                    save_checkpoint(state, os.path.join(workdir, 'best_ckpt'), np.round(best_val_acc, 2))
                except Exception as e:
                    print(e)

    if use_pmap:
        return flax.jax_utils.unreplicate(state)
    else:
        return state
