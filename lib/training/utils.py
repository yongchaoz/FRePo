import os
import logging
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state, checkpoints

from .metrics import pred_acurracy, top5_accuracy

Array = Any


class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.
    Attributes:
        batch_stats (Any): Collection used to store an exponential moving
                           average of the batch statistics.
        epoch (int): Current epoch.
        best_val_acc (float): Best validation accuracy
    """
    epoch: int
    best_val_acc: float
    batch_stats: Any = None
    ema_hidden: Any = None
    ema_average: Any = None
    ema_hidden_batch: Any = None
    ema_average_batch: Any = None
    ema_count: int = 0


@jax.jit
def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_util.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


@jax.jit
def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


class EMA():
    def __init__(self, decay, debias: bool = True):
        """Initializes an ExponentialMovingAverage module.

        References: https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
                    https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/moving_averages.py#L39#L137
        Args:
          decay: The chosen decay. Must in ``[0, 1)``. Values close to 1 result in
            slow decay; values close to ``0`` result in fast decay.
          debias: Whether to run with zero-debiasing.
        """
        self.decay = decay
        self.debias = debias
        self.hidden = None
        self.average = None
        self.count = None

    def initialize(self, state, hidden=None, average=None, count=None):
        if hidden is not None:
            assert average is not None, 'hidden and average should both be None or not None'
            self.hidden = hidden
            self.average = average
            self.count = count
        else:
            self.average = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state)
            self.hidden = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state)
            self.count = jnp.zeros([], jnp.int32)

    def __call__(self, value, update_stats: bool = True) -> jnp.ndarray:
        """Updates the EMA and returns the new value.
        Args:
          value: The array-like object for which you would like to perform an
            exponential decay on.
          update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When `update_stats` is False
            the internal stats will remain unchanged.
        Returns:
          The exponentially weighted average of the input value.
        """

        count = self.count + 1
        hidden = _update_moment(value, self.hidden, self.decay, order=1)

        average = hidden
        if self.debias:
            average = _bias_correction(hidden, self.decay, count)

        if update_stats:
            self.hidden = hidden
            self.average = average
            self.count = count

        return average

    @property
    def ema(self):
        return self.average


class AVG():
    def __init__(self):
        """Initializes an ExponentialMovingAverage module.

        References: https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
                    https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/moving_averages.py#L39#L137
        Args:
          decay: The chosen decay. Must in ``[0, 1)``. Values close to 1 result in
            slow decay; values close to ``0`` result in fast decay.
          debias: Whether to run with zero-debiasing.
        """
        self.hidden = None
        self.average = None
        self.count = None

    def initialize(self, state, hidden=None, average=None, count=None):
        if hidden is not None:
            assert average is not None, 'hidden and average should both be None or not None'
            self.hidden = hidden
            self.average = average
            self.count = count
        else:
            self.average = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state)
            self.hidden = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state)
            self.count = jnp.zeros([], jnp.int32)

        if not isinstance(self.hidden, dict):
            self.hidden = self.hidden.unfreeze()
        if not isinstance(self.average, dict):
            self.average = self.average.unfreeze()

    def __call__(self, value, update_stats: bool = True) -> jnp.ndarray:
        """Updates the EMA and returns the new value.
        Args:
          value: The array-like object for which you would like to perform an
            exponential decay on.
          update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When `update_stats` is False
            the internal stats will remain unchanged.
        Returns:
          The exponentially weighted average of the input value.
        """

        count = self.count + 1
        value = value.unfreeze()
        hidden = jax.tree_util.tree_map(
            lambda h, v: h * (self.count / count) + v / count, self.hidden, value)

        average = hidden

        if update_stats:
            self.hidden = hidden
            self.average = average
            self.count = count

        return average

    @property
    def avg(self):
        return self.average


def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.
    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.
    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    return checkpoints.restore_checkpoint(path, state)


def save_checkpoint(state, path, step_or_metric=None, keep=1):
    """
    Saves a checkpoint from the given state.
    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.
    """
    if jax.device_count() > 1:
        if jax.process_index() == 0:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        else:
            return

    if step_or_metric is None:
        checkpoints.save_checkpoint(path, state, int(state.step), keep=keep, overwrite=True)
    else:
        checkpoints.save_checkpoint(path, state, step_or_metric, keep=keep)


def make_chunky_prediction(x, predict_fn, chunk_size=1000):
    y = []
    for i in range(int(x.shape[0] / chunk_size)):
        x_chunk = x[i * chunk_size:(i + 1) * chunk_size]
        y_chunk = predict_fn(x_chunk)
        y.append(y_chunk)

    if x.shape[0] % chunk_size != 0:
        x_chunk = x[int(x.shape[0] / chunk_size) * chunk_size:]
        y_chunk = predict_fn(x_chunk)
        y.append(y_chunk)

    return np.vstack(y)


def process_batch(batch, use_pmap=False, dtype=jnp.float32, diff_aug=None, rng=None):
    if isinstance(batch, dict):
        image = batch['image'].astype(dtype)
        label = batch['label'].astype(dtype)
    elif isinstance(batch, Sequence):
        image = batch[0].astype(dtype)
        label = batch[1].astype(dtype)
    else:
        raise ValueError('Unknown Type {}'.format(type(batch)))

    if diff_aug is not None:
        image = diff_aug(rng, image)

    if use_pmap:
        num_devices = jax.device_count()
        # Reshape images from [num_devices * batch_size, height, width, img_channels]
        # to [num_devices, batch_size, height, width, img_channels].
        # The first dimension will be mapped across devices with jax.pmap.
        image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
        label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])
    return image, label


def compute_metrics(logits, labels, loss_type):
    """
    Computes the cross entropy loss and accuracy.
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B, num_classes] or [B,].
    Returns:
        (dict): Dictionary containing the cross entropy loss and accuracy.
    """
    loss = loss_type(logits, labels).mean()
    if labels.ndim == 2:
        labels = labels.argmax(1)
    accuracy = pred_acurracy(logits, labels).mean()
    top5accuracy = top5_accuracy(logits, labels).mean()

    metrics = {'loss': loss, 'accuracy': accuracy, 'top5accuracy': top5accuracy}
    return metrics


def initialized(key, img_size, img_channels, model, has_bn=False):
    """Initialize the model"""
    input_shape = (1, img_size, img_size, img_channels)

    @jax.jit
    def init(*args):
        return model.init(*args)

    key1, key2 = jax.random.split(key)
    variables = init({'params': key1, 'dropout': key2}, jnp.ones(input_shape, model.dtype))

    if has_bn:
        return variables['params'], variables['batch_stats']
    else:
        return variables['params']


def create_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs, warmup_epochs):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def create_train_state(rng, config, model, learning_rate_fn, has_bn=False, params=None, batch_stats=None, mask=None):
    """Create initial training state."""
    if config.optimizer == 'sgd':
        tx = optax.sgd(learning_rate=learning_rate_fn,
                       momentum=config.momentum,
                       nesterov=True)
    elif config.optimizer == 'adam':
        tx = optax.adam(learning_rate=learning_rate_fn)
    elif config.optimizer == 'adabelief':
        tx = optax.adam(learning_rate=learning_rate_fn)
    elif config.optimizer == 'adamw':
        tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=config.weight_decay, mask=mask)
    elif config.optimizer == 'lamb':
        tx = optax.lamb(learning_rate=learning_rate_fn, weight_decay=config.weight_decay)
    else:
        raise ValueError('Unknown Optimizer Type {}'.format(config.optimizer))

    if has_bn:
        if params is None:
            params, batch_stats = initialized(rng, config.img_size, config.img_channels, model, has_bn=True)

        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  ema_hidden=params,
                                  ema_average=params,
                                  batch_stats=batch_stats,
                                  ema_hidden_batch=batch_stats,
                                  ema_average_batch=batch_stats,
                                  epoch=0, best_val_acc=0.0)
    else:
        if params is None:
            params = initialized(
                rng, config.img_size, config.img_channels, model, has_bn=False)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  ema_hidden=params,
                                  ema_average=params,
                                  epoch=0, best_val_acc=0.0, )
    return state


def train_step(state, batch, rng, loss_type, l2_reg=0.0, has_feat=False, has_bn=False, use_pmap=True):
    def loss_fn(params):
        if has_bn:
            variables = {'params': params, 'batch_stats': state.batch_stats}
        else:
            variables = {'params': params}

        if has_feat:
            (logits, feat), new_model_state = state.apply_fn(variables, batch['image'], rngs={'dropout': rng},
                                                             train=True, mutable=['batch_stats'])
        else:
            logits, new_model_state = state.apply_fn(variables, batch['image'], rngs={'dropout': rng}, train=True,
                                                     mutable=['batch_stats'])

        loss = loss_type(logits, batch['label']).mean()
        if l2_reg > 0.0:
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_l2 = sum([jnp.sum(x ** 2)
                             for x in weight_penalty_params if x.ndim > 1])
            weight_penalty = l2_reg * 0.5 * weight_l2
            loss = loss + weight_penalty
        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    if jax.device_count() > 1 and use_pmap:
        grads = jax.lax.pmean(grads, axis_name='batch')

    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'], loss_type)

    if has_bn:
        new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def eval_step(state, batch, loss_type, has_feat=False, has_bn=False, use_ema=False):
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

    if has_feat:
        logits, feat = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    else:
        logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'], loss_type)


def pred_step(state, x, has_feat=False, has_bn=False, use_ema=False):
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

    if has_feat:
        logits, feat = state.apply_fn(variables, x, train=False, mutable=False)
    else:
        logits = state.apply_fn(variables, x, train=False, mutable=False)

    return logits


def save_logit(output_dir, state, x_train, x_test, chunk_size=500):
    @jax.jit
    def pred_fn(x): return pred_step(state, x)

    logit_train = make_chunky_prediction(x_train, pred_fn, chunk_size=chunk_size)
    logit_test = make_chunky_prediction(x_test, pred_fn, chunk_size=chunk_size)

    with open('{}/pred_logit.npz'.format(output_dir), 'wb') as f:
        np.savez(f, train=logit_train, test=logit_test)

    return logit_train, logit_test


def load_model_state(model, ckpt_path, config, key=0):
    state = create_train_state(jax.random.PRNGKey(key), config, model, lambda x: 0.01)
    if not os.path.exists(ckpt_path):
        raise ValueError('Checkpoint path {} does not exists!'.format(ckpt_path))
    state = checkpoints.restore_checkpoint(ckpt_path, state)
    return state


def load_random_state(model, config, key=0):
    state = create_train_state(jax.random.PRNGKey(key), config, model, lambda x: 0.01)
    return state


def load_teacher_model(model, ckpt_path, config):
    state = load_model_state(model, ckpt_path, config)

    @jax.jit
    def pred_fn(x): return pred_step(state, x)

    return pred_fn


def load_logit(output_dir):
    pred_logit = np.load('{}/pred_logit.npz'.format(output_dir))
    logging.info('Load logit from {}!'.format(output_dir))
    return pred_logit['train'], pred_logit['test']
