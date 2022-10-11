from typing import List, Tuple, Union, Iterable
from functools import wraps
import jax.numpy as jnp
import numpy as np

from jax import lax

__all__ = ["cutout", "channel_dropout"]


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return


@preserve_shape
def channel_dropout(
        img: jnp.ndarray, channels_to_drop: Union[int, Tuple[int, ...], jnp.ndarray], fill_value: Union[int, float] = 0
) -> jnp.ndarray:
    if len(img.shape) == 2 or img.shape[2] == 1:
        raise NotImplementedError("Only one channel. ChannelDropout is not defined.")

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


def cutout(
        img: jnp.ndarray, holes: Iterable[Tuple[int, int, int]], fill_value: Union[int, float] = 0, max_w_size: int = 8,
        max_h_size: int = 8
) -> jnp.ndarray:
    # Make a copy of the input image since we don't want to modify it directly
    mask = jnp.ones((max_w_size, max_h_size, img.shape[-1])) * fill_value

    for start_indices in holes:
        img = lax.dynamic_update_slice(img, mask, start_indices)
    return img

    # img = img.copy()
    # for x1, y1, x2, y2 in holes:
    #     img[y1:y2, x1:x2] = fill_value
    # return img
