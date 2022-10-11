import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


@jax.vmap
def top5_accuracy(logits, labels):
    val, idx = jax.lax.top_k(logits, k=5)
    return jnp.sum(idx == labels)

def pred_acurracy(logits, labels):
    """
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].
    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return jnp.argmax(logits, -1) == labels


def cross_entropy_loss(logits, labels):
    """
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B, num_classes].
    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    labels = jax.nn.one_hot(labels.argmax(-1), num_classes=labels.shape[-1])
    return -jnp.sum(labels * nn.log_softmax(logits), axis=-1)


def soft_cross_entropy_loss(logits, labels):
    """
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B, num_classes].
    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    labels = nn.softmax(labels)
    return -jnp.sum(labels * nn.log_softmax(logits), axis=-1)


def mean_squared_loss(logits, labels):
    """
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B, num_classes].
    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return jnp.sum((logits - labels) ** 2 * 0.5, axis=-1)


def logit_sum_loss(logits, labels):
    """
    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B, num_classes].
    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return jnp.sum(logits, axis=-1)


def stack_forest(forest):
    stack_args = lambda *args: np.stack(args)
    return jax.tree_util.tree_map(stack_args, *forest)


def get_metrics(device_metrics, use_pmap=False):
    if use_pmap:
        # We select the first element of x in order to get a single copy of a
        # device-replicated metric.
        device_metrics = jax.tree_util.tree_map(lambda x: x[0], device_metrics)

    metrics_np = jax.device_get(device_metrics)
    return stack_forest(metrics_np)
