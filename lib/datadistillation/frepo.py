import os
import time
import dataclasses
import functools

import tqdm
from absl import logging

import pandas as pd
import numpy as np
import tensorflow as tf

import jax
import jax.scipy as sp
import jax.numpy as jnp

import optax
import flax
import flax.linen as nn
from flax.training import checkpoints
from flax.training import train_state

from ..training.metrics import mean_squared_loss, get_metrics, top5_accuracy, soft_cross_entropy_loss, \
    cross_entropy_loss
from ..training.utils import compute_metrics, pred_acurracy, EMA
from ..training.utils import save_checkpoint, restore_checkpoint, process_batch, train_step, eval_step

from clu import metric_writers

from typing import Any, Iterable

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


class ProtoState(train_state.TrainState):
    epoch: int
    best_val_acc: float
    ema_hidden: Any = None
    ema_average: Any = None
    ema_count: int = 0


def init_proto(ds, num_prototypes_per_class, num_classes, class_subset=None, seed=0, scale_y=False):
    window_size = num_prototypes_per_class
    reduce_func = lambda key, dataset: dataset.batch(window_size)
    ds = ds.shuffle(num_prototypes_per_class * num_classes * 10, seed=seed)
    ds = ds.group_by_window(key_func=lambda x, y: y, reduce_func=reduce_func, window_size=window_size)

    if class_subset is None:
        is_init = [0] * num_classes
    else:
        is_init = [1] * num_classes
        for cls in class_subset:
            is_init[cls] = 0

    x_proto = [None] * num_classes
    y_proto = [None] * num_classes
    for ele in ds.as_numpy_iterator():
        cls = ele[1][0]
        if is_init[cls] == 1:
            pass
        else:
            x_proto[cls] = ele[0]
            y_proto[cls] = ele[1]
            is_init[cls] = 1
        if sum(is_init) == num_classes:
            break
    x_proto = np.concatenate([x for x in x_proto if x is not None], axis=0)
    y_proto = np.concatenate([y for y in y_proto if y is not None], axis=0)
    y_proto = jax.nn.one_hot(y_proto, num_classes)

    # center and scale y_proto
    y_proto = y_proto - 1 / num_classes
    if scale_y:
        y_scale = np.sqrt(num_classes / 10)
        y_proto = y_proto / y_scale
    return x_proto, y_proto


def create_proto_state(rng, model, learning_rate_fn, optimizer='lamb'):
    """Create initial training state."""
    if optimizer == 'lamb':
        tx = optax.lamb(learning_rate=learning_rate_fn)
    elif optimizer == 'adam':
        tx = optax.adam(learning_rate=learning_rate_fn)
    else:
        raise ValueError('Unknown Optimizer {}!'.format(optimizer))

    variables = model.init({'params': rng}).unfreeze()

    state = ProtoState.create(apply_fn=model.apply, tx=tx, params=variables['params'], epoch=0, best_val_acc=0.0)
    return state


def load_proto_state(model, ckpt_path):
    state = create_proto_state(jax.random.PRNGKey(0), model, lambda x: 0.01)
    if not os.path.exists(ckpt_path):
        raise ValueError('Checkpoint path {} does not exists!'.format(ckpt_path))
    state = checkpoints.restore_checkpoint(ckpt_path, state)
    return state


def nfr(x_target, x_proto, y_proto, nn_state, feat_fn, reg=1e-6):
    x_proto = feat_fn(x_proto, nn_state)
    k_pp = x_proto.dot(x_proto.T)
    k_tp = x_target.dot(x_proto.T)
    k_pp_reg = (k_pp + jnp.abs(reg) * jnp.trace(k_pp) * jnp.eye(k_pp.shape[0]) / k_pp.shape[0])
    pred = jnp.dot(k_tp, sp.linalg.solve(k_pp_reg, y_proto, sym_pos=True))
    return pred


def nn_feat_fn(x, state, has_bn=False, use_ema=False):
    if use_ema:
        if has_bn:
            variables = {'params': state.ema_average, 'batch_stats': state.ema_average_batch}
        else:
            variables = {'params': state.ema_average}
    else:
        if has_bn:
            variables = {'params': state.params, 'batch_stats': state.batch_stats}
        else:
            variables = {'params': state.params}
    return state.apply_fn(variables, x, train=False, mutable=False)[1]


class ProtoHolder(nn.Module):
    x_proto_init: Array
    y_proto_init: Array
    num_prototypes: int
    learn_label: bool = True

    @nn.compact
    def __call__(self, ):
        x_proto = self.param('x_proto', lambda *_: self.x_proto_init)
        y_proto = self.param('y_proto', lambda *_: self.y_proto_init)

        if not self.learn_label:
            y_proto = jax.lax.stop_gradient(y_proto)

        return x_proto, y_proto


@jax.vmap
def lb_margin(logits):
    sorted_logits = jnp.sort(logits)
    return -(sorted_logits[-1] - sorted_logits[-2])


@jax.vmap
def lb_margin_th(logits):
    dim = logits.shape[-1]
    val, idx = jax.lax.top_k(logits, k=2)
    margin = jnp.minimum(val[0] - val[1], 1 / dim)
    return -margin


def lb_l2(y_proto, y_proto_init):
    loss = jnp.sum((y_proto - y_proto_init) ** 2, axis=-1)
    return loss


def proto_train_step(state, nn_state, batch, use_flip=False, feat_fn=None):
    def loss_fn(params, images, labels):
        x_proto, y_proto = state.apply_fn(variables={'params': params})
        lb_loss = lb_margin_th(y_proto).mean()

        if use_flip:
            x_proto_flip = jnp.flip(x_proto, axis=-2)
            x_proto = jnp.concatenate([x_proto, x_proto_flip], axis=0)
            y_proto = jnp.concatenate([y_proto, y_proto], axis=0)

        preds = nfr(images, x_proto, y_proto, nn_state, feat_fn)
        accuracy = pred_acurracy(preds, labels.argmax(-1)).mean()
        top5accuracy = top5_accuracy(preds, labels.argmax(-1)).mean()
        kernel_loss = mean_squared_loss(preds, labels).mean()
        loss = kernel_loss + lb_loss

        return loss, (accuracy, top5accuracy, kernel_loss, lb_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params, batch['image'], batch['label'])

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    if jax.device_count() > 1:
        grads = jax.lax.pmean(grads, axis_name='batch')

    grad_x = jnp.concatenate([v.reshape(1, -1) for k, v in grads.items() if 'x_proto' in k], axis=0)
    grad_y = jnp.concatenate([v.reshape(1, -1) for k, v in grads.items() if 'y_proto' in k], axis=0)
    grad_norm_x = jnp.linalg.norm(grad_x, ord=2, axis=-1)
    grad_norm_y = jnp.linalg.norm(grad_y, ord=2, axis=-1)

    accuracy, top5accuracy, kernel_loss, lb_loss = aux[1]

    metrics = {'total_loss': aux[0], 'accuracy': accuracy, 'top5accuracy': top5accuracy, 'kernel_loss': kernel_loss,
               'label_loss': lb_loss, 'grad_norm_x': jnp.mean(grad_norm_x), 'grad_norm_y': jnp.mean(grad_norm_y)}

    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def proto_eval_step(x_proto, y_proto, k_pp_reg, nn_state, batch, feat_fn=None):
    x_target = feat_fn(batch['image'], nn_state)
    k_tp = x_target.dot(x_proto.T)
    pred = jnp.dot(k_tp, sp.linalg.solve(k_pp_reg, y_proto, sym_pos=True))
    return compute_metrics(pred, batch['label'], mean_squared_loss)


def compute_feat_kpp(x_proto, nn_state, feat_fn=None, reg=1e-6, chunk_size=500):
    train_size = x_proto.shape[0]
    if train_size <= chunk_size:
        x_proto = feat_fn(x_proto, nn_state)
    else:
        x_proto_list = []
        for i in range(train_size // chunk_size):
            x_proto_list.append(feat_fn(x_proto[i * chunk_size:(i + 1) * chunk_size], nn_state))

        if train_size % chunk_size != 0:
            x_proto_list.append(feat_fn(x_proto[train_size // chunk_size * chunk_size:], nn_state))
        x_proto = jnp.concatenate(x_proto_list, axis=0)

    k_pp = x_proto.dot(x_proto.T)
    k_pp_reg = (k_pp + jnp.abs(reg) * jnp.trace(k_pp) * jnp.eye(k_pp.shape[0]) / k_pp.shape[0])
    return x_proto, k_pp_reg


def jit_fn(nn_train_step, nn_eval_step, feat_fn, feat_fn_eval, use_flip=False, use_pmap=False):
    ln_train_step = functools.partial(proto_train_step, feat_fn=feat_fn, use_flip=use_flip)
    ln_eval_step = functools.partial(proto_eval_step, feat_fn=feat_fn_eval)

    if use_pmap:
        jit_train_step = jax.pmap(ln_train_step, axis_name='batch', in_axes=(0, 0, 0))  # state, nn_state, batch
        jit_eval_step = jax.pmap(ln_eval_step, axis_name='batch',
                                 in_axes=(0, None, 0, 0, None))  # x_proto, y_proto, k_pp_reg, nn_state, batch
        jit_nn_train_step = jax.pmap(nn_train_step, axis_name='batch', in_axes=(0, None, None))  # state, batch, rng
        jit_nn_eval_step = jax.pmap(nn_eval_step, axis_name='batch', in_axes=(0, None))  # state, batch
        jit_nn_train_step_online = jax.pmap(nn_train_step, axis_name='batch',
                                            in_axes=(0, None, None))  # state, batch, rng
    else:
        jit_train_step = jax.jit(ln_train_step)
        jit_eval_step = jax.jit(ln_eval_step)
        jit_nn_train_step = jax.jit(nn_train_step)
        jit_nn_eval_step = jax.jit(nn_eval_step)
        jit_nn_train_step_online = jax.jit(nn_train_step)

    return jit_train_step, jit_eval_step, jit_nn_train_step, jit_nn_eval_step, jit_nn_train_step_online


def train_on_proto(ds, nn_state, jit_nn_train_step, diff_aug, rng, num_updates=1000, has_bn=False):
    EMAState = EMA(decay=0.995, debias=True)
    EMAState.initialize(nn_state.params)

    if has_bn:
        EMABatch = EMA(decay=0.9, debias=True)
        EMABatch.initialize(nn_state.batch_stats)

    for step, (img, lb) in tqdm.tqdm(zip(range(num_updates), ds.as_numpy_iterator()), desc='Train on distilled data'):
        rng, train_rng = jax.random.split(rng)
        img = diff_aug(train_rng, img)
        nn_state, _ = jit_nn_train_step(nn_state, {'image': img, 'label': lb}, train_rng)

        # For simplicity, we do not update EMA count since we don't need to save model
        EMAState(nn_state.params)
        if has_bn:
            EMABatch(nn_state.batch_stats)

    nn_state = dataclasses.replace(nn_state, **{'ema_hidden': EMAState.hidden,
                                                'ema_average': EMAState.average})
    if has_bn:
        EMABatch(nn_state.batch_stats)
        nn_state = dataclasses.replace(nn_state, **{'ema_hidden_batch': EMABatch.hidden,
                                                    'ema_average_batch': EMABatch.average})
    return nn_state


def eval_on_proto_nn(ds_test, nn_state, jit_nn_eval_step, writer=None, step=0, name=''):
    nn_eval_metrics = []

    for batch in tqdm.tqdm(ds_test.as_numpy_iterator(), desc='Evaluate using nn predictor'):
        img, lb = process_batch(batch, use_pmap=False)
        metrics = jit_nn_eval_step(nn_state, {'image': img, 'label': lb})
        nn_eval_metrics.append(metrics)

    nn_eval_metrics = get_metrics(nn_eval_metrics, use_pmap=False)  # (num_eval, num_device)
    nn_summary_mean = jax.tree_map(lambda x: x.mean(), nn_eval_metrics)  # mean of num_device models
    nn_summary_std = jax.tree_map(lambda x: x.mean(0).std(), nn_eval_metrics)  # std of num_device models

    if writer is not None:
        writer.write_scalars(step + 1, {f'{name}/nn_{k}_mean': v for k, v in nn_summary_mean.items()})
        writer.write_scalars(step + 1, {f'{name}/nn_{k}_std': v for k, v in nn_summary_std.items()})
        writer.flush()
    else:
        logging.info(str({f'{name}/nn_{k}_mean': v for k, v in nn_summary_mean.items()}))

    return nn_summary_mean


def eval_on_proto_krr(ds_test, x_proto, y_proto, nn_state, jit_eval_step, jit_compute_feat_kpp, use_flip=False,
                      writer=None, step=0, name=''):
    if use_flip:
        x_proto_flip = jnp.flip(x_proto, axis=-2)
        x_proto = jnp.concatenate([x_proto, x_proto_flip], axis=0)
        y_proto = jnp.concatenate([y_proto, y_proto], axis=0)

    x_proto, k_pp_reg = jit_compute_feat_kpp(x_proto, nn_state)

    krr_eval_metrics = []
    for batch in tqdm.tqdm(ds_test.as_numpy_iterator(), desc='Evaluate using krr predictor'):
        img, lb = process_batch(batch, use_pmap=False)
        metrics = jit_eval_step(x_proto, y_proto, k_pp_reg, nn_state, {'image': img, 'label': lb})
        krr_eval_metrics.append(metrics)

    krr_eval_metrics = get_metrics(krr_eval_metrics, use_pmap=False)  # (num_eval, num_device)
    krr_summary_mean = jax.tree_map(lambda x: x.mean(), krr_eval_metrics)  # mean of num_device models
    krr_summary_std = jax.tree_map(lambda x: x.mean(0).std(), krr_eval_metrics)  # std of num_device models

    if writer is not None:
        writer.write_scalars(step + 1, {f'{name}/ln_{k}_mean': v for k, v in krr_summary_mean.items()})
        writer.write_scalars(step + 1, {f'{name}/ln_{k}_std': v for k, v in krr_summary_std.items()})
        writer.flush()
    else:
        logging.info(str({f'{name}/nn_{k}_mean': v for k, v in krr_summary_mean.items()}))

    return krr_summary_mean


def proto_eval(x_proto, y_proto, nn_state_eval, rng, ds_test, num_online_eval_updates,
               jit_nn_train_step, jit_nn_eval_step, diff_aug=None):
    ds_proto = tf.data.Dataset.from_tensor_slices((x_proto, y_proto))
    ds_proto = ds_proto.cache().repeat().shuffle(buffer_size=5000).batch(batch_size=min(y_proto.shape[0], 500))

    nn_state_eval = train_on_proto(ds_proto, nn_state_eval, jit_nn_train_step, diff_aug, rng,
                                   num_updates=num_online_eval_updates)

    nn_summary_mean = eval_on_proto_nn(ds_test, nn_state_eval, jit_nn_eval_step)

    return nn_summary_mean['accuracy'] * 100


def get_proto(state, use_pmap):
    if use_pmap:
        state = flax.jax_utils.unreplicate(state)

    x_proto, y_proto = state.apply_fn(variables={'params': state.params})
    return x_proto, y_proto


def get_sample_proto(state, rng, bs=500, use_pmap=False):
    x_proto, y_proto = get_proto(state, use_pmap)

    if y_proto.shape[0] < bs:
        img, lb = x_proto, y_proto
    else:
        sample_idx = jax.random.choice(rng, y_proto.shape[0], shape=(bs,), replace=False)
        img, lb = x_proto[sample_idx], y_proto[sample_idx]
    return img, lb


def proto_train_and_evaluate(config, ph, create_online_state, create_eval_state, dataset, workdir,
                             seed=0, num_nn_state=1, num_train_steps=None, use_flip=False,
                             num_online_eval_updates=100, diff_aug=None,
                             max_online_updates=50, steps_per_epoch=None, steps_per_log=None, steps_per_eval=None,
                             steps_per_checkpoint=None, save_ckpt=None, writer=None, image_saver=None,
                             steps_per_save_image=None, has_bn=False, resume=True, use_pmap=False, dtype=jnp.float32,
                             num_eval=5):
    # --------------------------------------
    # Setup
    # --------------------------------------
    ds_train, ds_test = dataset

    if writer is None:
        writer = metric_writers.create_default_writer(logdir=workdir)

    if steps_per_checkpoint is None:
        steps_per_checkpoint = steps_per_epoch * 5
    if steps_per_eval is None:
        steps_per_eval = steps_per_epoch * 10
    if steps_per_save_image is None:
        steps_per_save_image = steps_per_epoch * 250

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # --------------------------------------
    # Initialize Model and Optimizer
    # --------------------------------------
    jit_create_online_state = jax.jit(create_online_state)
    jit_create_eval_state = jax.jit(create_eval_state)
    jit_vmap_create_eval_state = jax.jit(jax.vmap(create_eval_state))
    jit_vmap_create_online_state = jax.jit(jax.vmap(create_online_state))
    jit_get_proto = jax.jit(functools.partial(get_proto, use_pmap=use_pmap))
    jit_get_sample_proto = jax.jit(functools.partial(get_sample_proto, use_pmap=use_pmap, bs=config.eval_batch_size))

    rng, init_rng = jax.random.split(rng)

    learning_rate_fn = optax.cosine_decay_schedule(init_value=config.learning_rate, decay_steps=num_train_steps,
                                                   alpha=0.1)
    state = create_proto_state(rng, ph, learning_rate_fn, optimizer=config.optimizer)

    # --------------------------------------
    # Initialize the model pool
    # --------------------------------------
    nn_online_states = []
    for i in range(num_nn_state):
        rng, init_rng = jax.random.split(rng)
        init_step = (max_online_updates // num_nn_state) * i
        if use_pmap:
            nn_state = jit_vmap_create_online_state(jax.random.split(init_rng, jax.device_count()))
            nn_state = dataclasses.replace(nn_state, **{'step': flax.jax_utils.replicate(init_step)})
        else:
            nn_state = jit_create_online_state(init_rng)
            nn_state = dataclasses.replace(nn_state, **{'step': init_step})
        nn_online_states.append(nn_state)

    idx = np.random.randint(low=0, high=num_nn_state)
    nn_state = nn_online_states[idx]

    feat_fn = functools.partial(nn_feat_fn, has_bn=has_bn, use_ema=False)
    feat_fn_eval = functools.partial(nn_feat_fn, has_bn=False, use_ema=False)

    nn_train_step = functools.partial(train_step, has_feat=True, has_bn=has_bn, loss_type=mean_squared_loss)
    nn_eval_step = functools.partial(eval_step, has_feat=True, has_bn=has_bn, use_ema=True, loss_type=mean_squared_loss)
    get_jit_fn = functools.partial(jit_fn, feat_fn=feat_fn, feat_fn_eval=feat_fn_eval, use_flip=use_flip,
                                   use_pmap=use_pmap)
    jit_train_step, jit_eval_step, jit_nn_train_step, jit_nn_eval_step, jit_nn_train_step_online = get_jit_fn(
        nn_train_step, nn_eval_step)

    nn_train_step_eval = functools.partial(train_step, has_feat=True, has_bn=False, loss_type=mean_squared_loss)
    nn_eval_step_eval = functools.partial(eval_step, has_feat=True, has_bn=False, loss_type=mean_squared_loss)
    jit_train_step_eval, jit_eval_step_eval, jit_nn_train_step_eval, jit_nn_eval_step_eval, _ = get_jit_fn(
        nn_train_step_eval, nn_eval_step_eval)

    if use_pmap:
        jit_feat_fn = jax.pmap(feat_fn, in_axes=(0, 0))
    else:
        jit_feat_fn = jax.jit(feat_fn)

    evaluate_proto = functools.partial(proto_eval, ds_test=ds_test, num_online_eval_updates=num_online_eval_updates,
                                       jit_nn_train_step=jit_nn_train_step_eval, diff_aug=diff_aug,
                                       jit_nn_eval_step=jit_nn_eval_step_eval)

    # --------------------------------------
    # Training
    # --------------------------------------
    step, step_offset, best_val_acc, accuracy = 0, 0, 0.0, 0.0
    log_steps = [1, 3000, 10000, 30000, 100000]

    if resume:
        state = restore_checkpoint(state, os.path.join(workdir, 'proto'))
        step_offset = jax.device_get(state.step)
        best_val_acc = jax.device_get(state.best_val_acc)

    if use_pmap:
        state = flax.jax_utils.replicate(state)

    train_metrics = []

    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_train_steps), ds_train.as_numpy_iterator()):
        # --------------------------------------
        # Train image
        # --------------------------------------
        rng, train_rng = jax.random.split(rng)

        image, label = process_batch(batch, use_pmap=use_pmap, dtype=dtype)
        feat = jit_feat_fn(image, nn_state)
        state, metrics = jit_train_step(state, nn_state, {'image': feat, 'label': label})

        train_metrics.append(metrics)

        if step == step_offset:
            logging.info('Initial compilation completed. Elapsed: {}s'.format(time.time() - train_metrics_last_t))
            train_metrics_last_t = time.time()

        if (step + 1) in log_steps or (step + 1) % steps_per_log == 0:
            train_metrics = get_metrics(train_metrics, use_pmap=use_pmap)

            summary = {f'train/{k}': v for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()}
            summary['monitor/steps_per_second'] = steps_per_log / (time.time() - train_metrics_last_t)
            summary['monitor/learning_rate'] = learning_rate_fn(step)

            x_proto, y_proto = jit_get_proto(state)
            x_proto_norm = jnp.linalg.norm(x_proto.reshape(x_proto.shape[0], -1), ord=2, axis=-1).mean()
            y_proto_norm = jnp.linalg.norm(y_proto, ord=2, axis=-1).mean()
            y_proto_max = y_proto.max(-1)
            y_proto_margin = -lb_margin(y_proto)

            summary['proto/x_proto_norm'] = x_proto_norm
            summary['proto/y_proto_norm'] = y_proto_norm
            summary['proto/y_proto_max_mean'] = y_proto_max.mean()
            summary['proto/y_proto_max_max'] = y_proto_max.max()
            summary['proto/y_proto_max_min'] = y_proto_max.min()
            summary['proto/y_proto_margin_mean'] = y_proto_margin.mean()
            summary['proto/y_proto_margin_max'] = y_proto_margin.max()
            summary['proto/y_proto_margin_min'] = y_proto_margin.min()

            writer.write_scalars(step + 1, summary)
            train_metrics = []
            train_metrics_last_t = time.time()

        # --------------------------------------
        # Update the model and sample a new model
        # --------------------------------------
        nn_step = nn_state.step[0] if use_pmap else nn_state.step

        if nn_step > max_online_updates:
            rng, init_rng = jax.random.split(rng)
            if use_pmap:
                init_rng = jax.random.split(init_rng, jax.device_count())
                nn_state = jit_vmap_create_online_state(init_rng)
            else:
                nn_state = jit_create_online_state(init_rng)
        else:
            rng, train_rng = jax.random.split(rng)
            img, lb = jit_get_sample_proto(state, train_rng)
            nn_state, _ = jit_nn_train_step_online(nn_state, {'image': img, 'label': lb}, train_rng)

        nn_online_states[idx] = nn_state
        idx = np.random.randint(low=0, high=num_nn_state)
        nn_state = nn_online_states[idx]

        # --------------------------------------
        # Evaluate a random model trained on distilled image using least norm predictor or nn
        # --------------------------------------
        if (step + 1) in log_steps or (step + 1) % (steps_per_eval) == 0:
            x_proto, y_proto = jit_get_proto(state)
            step_acc = []
            for i in range(num_eval):
                rng, init_rng = jax.random.split(rng)

                if use_pmap:
                    eval_rng = jax.random.split(init_rng, jax.device_count())
                    nn_state_eval = jit_vmap_create_eval_state(eval_rng)
                else:
                    nn_state_eval = jit_create_eval_state(init_rng)

                accuracy = evaluate_proto(x_proto, y_proto, nn_state_eval, rng)

                step_acc.append(accuracy)

            accuracy = np.mean(step_acc)
            writer.write_scalars(step + 1, {f'eval/step_acc_mean': np.mean(step_acc)})
            writer.write_scalars(step + 1, {f'eval/step_std': np.std(step_acc)})
            writer.flush()

        # --------------------------------------
        # Save state and visualization
        # --------------------------------------
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_train_steps:
            epoch = (step + 1) // steps_per_epoch
            try:
                save_checkpoint(state, os.path.join(workdir, 'proto'), step + 1)
            except Exception as e:
                print(e)

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
                    save_checkpoint(state, os.path.join(workdir, 'best_ckpt'), np.round(best_val_acc, 2), keep=5)
                except Exception as e:
                    print(e)

        if (step + 1) in log_steps or (step + 1) % save_ckpt == 0:
            try:
                save_checkpoint(state, os.path.join(workdir, 'saved_ckpt'), step + 1, keep=15)
            except Exception as e:
                print(e)

        if image_saver:
            if (step + 1) in log_steps or (step + 1) % steps_per_save_image == 0 or (step + 1) == num_train_steps:
                image_saver(state, step=step + 1, use_pmap=use_pmap)

    if use_pmap:
        return flax.jax_utils.unreplicate(state)
    else:
        return state


def proto_evaluate(ds_train, ds_test, workdir, create_eval_state, num_online_eval_updates=100, loss='mse',
                   seed=0, use_flip=False, diff_aug=None, steps_per_log=None, writer=None, has_bn=False,
                   use_pmap=False, name='', x_proto=None, y_proto=None, eval_krr=False, num_eval=5):
    # --------------------------------------
    # Setup
    # --------------------------------------
    if loss == 'mse':
        loss_type = mean_squared_loss
    elif loss == 'xent':
        loss_type = cross_entropy_loss
    elif loss == 'softxent':
        loss_type = soft_cross_entropy_loss
    else:
        raise ValueError('Unknown loss function: {}'.format(loss))

    if writer is None:
        writer = metric_writers.create_default_writer(logdir=workdir)

    jit_create_eval_state = jax.jit(create_eval_state)
    jit_vmap_create_eval_state = jax.jit(jax.vmap(create_eval_state))

    feat_fn = functools.partial(nn_feat_fn, has_bn=has_bn, use_ema=False)
    feat_fn_eval = functools.partial(nn_feat_fn, has_bn=has_bn, use_ema=False)

    get_jit_fn = functools.partial(jit_fn, feat_fn=feat_fn, feat_fn_eval=feat_fn_eval, use_flip=use_flip,
                                   use_pmap=use_pmap)

    nn_train_step = functools.partial(train_step, has_feat=True, has_bn=has_bn, loss_type=loss_type)
    nn_eval_step = functools.partial(eval_step, has_feat=True, has_bn=has_bn, loss_type=loss_type)
    jit_train_step, jit_eval_step, jit_nn_train_step, jit_nn_eval_step, _ = get_jit_fn(
        nn_train_step, nn_eval_step)

    if use_pmap:
        jit_compute_feat_kpp = jax.pmap(functools.partial(compute_feat_kpp, feat_fn=feat_fn_eval),
                                        in_axes=(None, 0))  # x_proto, nn_state
    else:
        jit_compute_feat_kpp = jax.jit(functools.partial(compute_feat_kpp, feat_fn=feat_fn_eval))

    # --------------------------------------
    # Train on proto and evaluate
    # --------------------------------------
    if use_pmap:
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        eval_rng = jax.random.split(init_rng, jax.device_count())
        nn_state = jit_vmap_create_eval_state(eval_rng)
        train_metrics = []
        nn_acc_mean = 0

        for step, (img, lb) in tqdm.tqdm(zip(range(num_online_eval_updates), ds_train.as_numpy_iterator()),
                                         desc='Train on distilled data'):
            rng, train_rng = jax.random.split(rng)
            img = diff_aug(train_rng, img)
            nn_state, metrics = jit_nn_train_step(nn_state, {'image': img, 'label': lb}, train_rng)
            train_metrics.append(metrics)

            if (step + 1) % steps_per_log == 0:
                nn_train_metrics = get_metrics(train_metrics, use_pmap=False)  # (num_eval, num_device)
                nn_summary_mean = jax.tree_map(lambda x: x.mean(), nn_train_metrics)  # mean of num_device models
                nn_summary_std = jax.tree_map(lambda x: x.mean(0).std(), nn_train_metrics)  # std of num_device models

                if writer is None:
                    logging.info('Step {}:'.format(step) + str(nn_summary_mean))
                else:
                    if name != "":
                        writer.write_scalars(step + 1,
                                             {f'train/{name}/nn_{k}_mean': v for k, v in nn_summary_mean.items()})
                        writer.write_scalars(step + 1,
                                             {f'train/{name}/nn_{k}_std': v for k, v in nn_summary_std.items()})
                    else:
                        writer.write_scalars(step + 1, {f'train/nn_{k}_mean': v for k, v in nn_summary_mean.items()})
                        writer.write_scalars(step + 1, {f'train/nn_{k}_std': v for k, v in nn_summary_std.items()})
                    writer.flush()
                train_metrics = []

                nn_acc_mean = eval_on_proto_nn(ds_test, nn_state, jit_nn_eval_step, writer, step,
                                               name=f'eval_nn/{name}' if name != "" else 'eval')

                if eval_krr:
                    krr_acc_mean = eval_on_proto_krr(ds_test, x_proto, y_proto, nn_state, jit_eval_step,
                                                     jit_compute_feat_kpp, use_flip, writer, step,
                                                     name=f'eval_krr/{name}' if name != "" else 'eval')

        return nn_state, nn_acc_mean['accuracy']
    else:
        rng = jax.random.PRNGKey(seed)
        acc_list = []
        nn_state = None
        for i in range(num_eval):
            logging.info(f"=========== Model {i} ============")
            rng, init_rng = jax.random.split(rng)
            nn_state = jit_create_eval_state(init_rng)

            train_metrics = []
            nn_acc_mean = 0

            for step, (img, lb) in tqdm.tqdm(zip(range(num_online_eval_updates), ds_train.as_numpy_iterator()),
                                             desc='Train on distilled data'):
                rng, train_rng = jax.random.split(rng)
                img = diff_aug(train_rng, img)
                nn_state, metrics = jit_nn_train_step(nn_state, {'image': img, 'label': lb}, train_rng)
                train_metrics.append(metrics)

                if (step + 1) % steps_per_log == 0:
                    nn_train_metrics = get_metrics(train_metrics, use_pmap=False)  # (num_eval, num_device)
                    nn_summary_mean = jax.tree_map(lambda x: x.mean(), nn_train_metrics)  # mean of num_device models

                    if writer is None:
                        logging.info('Step {}:'.format(step) + str(nn_summary_mean))
                    else:
                        if name != "":
                            writer.write_scalars(step + 1,
                                                 {f'train/{name}/nn_{k}_mean_{i}': v for k, v in nn_summary_mean.items()})
                        else:
                            writer.write_scalars(step + 1,
                                                 {f'train/nn_{k}_mean_{i}': v for k, v in nn_summary_mean.items()})
                        writer.flush()
                    train_metrics = []

                    nn_acc_mean = eval_on_proto_nn(ds_test, nn_state, jit_nn_eval_step, writer, step,
                                                   name=f'eval_nn/{name}_{i}' if name != "" else 'eval')

                    if eval_krr:
                        krr_acc_mean = eval_on_proto_krr(ds_test, x_proto, y_proto, nn_state, jit_eval_step,
                                                         jit_compute_feat_kpp, use_flip, writer, step,
                                                         name=f'eval_krr/{name}_{i}' if name != "" else 'eval')
            acc_list.append(nn_acc_mean['accuracy'])
        logging.info(f'Evaluate on {num_eval} models, mean: {np.mean(acc_list)}, std: {np.std(acc_list)}')

        return nn_state, acc_list
