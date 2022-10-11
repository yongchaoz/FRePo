# Copyright 2021 Konrad Heidler
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from typing import Union, List, Tuple

import math
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
import warnings

from .base import Transformation, InputType, same_type
from .utils import log_uniform
from .functional.dropout import cutout


class ImageLevelTransformation(Transformation):
    pass


class GridShuffle(ImageLevelTransformation):
    """Divides the image into grid cells and shuffles them randomly.

    Args:
        grid_size (int, int): Tuple of `(gridcells_x, gridcells_y)` that specifies into how many
            cells the image is to be divided along each axis.
            If only a single number is given, that value will be used along both axes.
            Currently requires that each image dimension is a multiple of the corresponding value.
        p (float): Probability of applying the transformation
    """

    def __init__(self, grid_size: Union[Tuple[int, int], int] = (4, 4), p: float = 0.5, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        if hasattr(grid_size, '__iter__'):
            self.grid_size = tuple(grid_size)
        else:
            self.grid_size = (self.grid_size, self.grid_size)
        self.grid_size = grid_size
        self.probability = p

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType] = None, invert=False) -> List[
        jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        key1, key2 = jax.random.split(rng)
        do_apply = jax.random.bernoulli(key1, self.probability)
        val = []
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE) or same_type(type, InputType.MASK) or same_type(type, InputType.DENSE):
                raw_image = input

                H, W, C = raw_image.shape
                gx, gy = self.grid_size

                if H % self.grid_size[0] != 0:
                    raise ValueError(f"Image height ({H}) needs to be a multiple of gridcells_y ({gy})")
                if W % self.grid_size[1] != 0:
                    raise ValueError(f"Image width ({W}) needs to be a multiple of gridcells_x ({gx})")

                image = rearrange(raw_image, '(gy h) (gx w) c -> (gy gx) h w c', gx=gx, gy=gy)
                if invert:
                    inv_permutation = jnp.argsort(jax.random.permutation(key2, image.shape[0]))
                    image = image[inv_permutation]
                else:
                    image = jax.random.permutation(key2, image)
                image = rearrange(image, '(gy gx) h w c -> (gy h) (gx w) c', gx=gx, gy=gy)
                current = jnp.where(do_apply, image, raw_image)
            else:
                raise NotImplementedError(f"GridShuffle for {type} not yet implemented")
                current = input
            val.append(current)
        return val


class _ConvolutionalBlur(ImageLevelTransformation):
    @abstractmethod
    def __init__(self, p: float = 0.5, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        self.probability = p
        self.kernel = None
        self.kernelsize = -1

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType] = None, invert=False) -> List[
        jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        val = []
        do_apply = jax.random.bernoulli(rng, self.probability)
        p0 = self.kernelsize // 2
        p1 = self.kernelsize - p0 - 1
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE):
                if invert:
                    warnings.warn("Trying to invert a Blur Filter, which is not invertible.")
                    current = input
                else:
                    image_padded = jnp.pad(input, [(p0, p1), (p0, p1), (0, 0)], mode='edge')
                    image_padded = rearrange(image_padded, 'h w (c c2) -> c c2 h w', c2=1)
                    convolved = jax.lax.conv(image_padded, self.kernel, [1, 1], 'valid')
                    convolved = rearrange(convolved, 'c c2 h w -> h w (c c2)', c2=1)
                    current = jnp.where(do_apply, convolved, input)
            else:
                current = input
            val.append(current)
        return val


class Blur(_ConvolutionalBlur):
    def __init__(self, size: int = 5, p: float = 0.5):
        super().__init__(p)
        self.kernel = jnp.ones([1, 1, size, size])
        self.kernel = self.kernel / self.kernel.sum()
        self.kernelsize = size


class GaussianBlur(_ConvolutionalBlur):
    def __init__(self, sigma: int = 3, p: float = 0.5):
        super().__init__(p)
        N = int(math.ceil(2 * sigma))
        rng = jnp.linspace(-2.0, 2.0, N)
        x = rng.reshape(1, -1)
        y = rng.reshape(-1, 1)

        self.kernel = jnp.exp((-0.5 / sigma) * (x * x + y * y))
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel.reshape(1, 1, N, N)
        self.kernelsize = N


class Cutout(ImageLevelTransformation):
    """CoarseDropout of the square regions in the image.
    Args:
        num_holes (int): number of regions to zero out
        max_h_size (int): maximum height of the hole
        max_w_size (int): maximum width of the hole
        fill_value (int, float, list of int, list of float): value for dropped pixels.
    Targets:
        image
    Image types:
        uint8, float32
    Reference:
    |  https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/dropout/cutout.py
    """

    def __init__(self, num_holes: int = 8,
                 max_h_size: int = 8,
                 max_w_size: int = 8,
                 fill_value: Union[int, float] = 0,
                 p: float = 0.5):
        super().__init__()
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.probability = p

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType] = None, invert=False) -> List[
        jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        key1, key2 = jax.random.split(rng)
        do_apply = jax.random.bernoulli(key1, self.probability)
        val = []

        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE) or same_type(type, InputType.MASK) or same_type(type, InputType.DENSE):
                raw_image = input

                H, W, C = raw_image.shape

                holes = []
                for _ in range(self.num_holes):
                    new_rng, key2 = jax.random.split(key2)
                    y, x = jax.random.randint(new_rng, [2], minval=jnp.array([0, 0]),
                                              maxval=jnp.array([H - self.max_h_size+1, W - self.max_w_size+1]))
                    holes.append((x, y, 0))

                image = cutout(raw_image, holes, self.fill_value, self.max_w_size, self.max_h_size)

                if invert:
                    warnings.warn("Trying to invert a cutout image, which is not invertible.")
                    current = raw_image
                else:
                    current = jnp.where(do_apply, image, raw_image)
            else:
                raise NotImplementedError(f"Cutout for {type} not yet implemented")
            val.append(current)
        return val


class NormalizedColorJitter(ImageLevelTransformation):
    """Randomly jitter the image colors when the image is normalized.

    Args:
        range (float, float):
        p (float): Probability of applying the transformation

    Reference: https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py
    """

    def __init__(self,
                 brightness: float = 0.5,
                 contrast: float = 1.0,
                 saturation: float = 0.5,
                 p: float = 0.5,
                 input_types=None
                 ):
        super().__init__(input_types)
        self.brightness = brightness
        self.contrast = np.exp(contrast) if contrast > 0 else 0.0
        self.saturation = np.exp(saturation) if saturation > 0 else 0.0
        self.probability = p

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType] = None, invert=False) -> List[
        jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        keys = jax.random.split(rng, 3)
        val = []

        for input, type in zip(inputs, input_types):
            if same_type(type, InputType.IMAGE) or same_type(type, InputType.MASK) or same_type(type, InputType.DENSE):
                x = input
                ops = ['brightness', 'contrast', 'saturation']

                for op, key in zip(ops, keys):
                    strength = getattr(self, op)
                    if strength <= 0:
                        continue
                    if op == 'brightness':
                        randb = jax.random.uniform(key, minval=-self.brightness, maxval=self.brightness)
                        x_new = x + randb
                    elif op == 'contrast':
                        randc = log_uniform(key, minval=1 / self.contrast, maxval=self.contrast)
                        x_mean = x.mean(axis=(-1, -2, -3), keepdims=True)
                        x_new = (x - x_mean) * randc + x_mean
                    elif op == 'saturation':
                        rands = log_uniform(key, minval=1 / self.saturation, maxval=self.saturation)
                        x_mean = x.mean(axis=-1, keepdims=True)
                        x_new = (x - x_mean) * rands + x_mean
                    else:
                        raise ValueError('Unknown operation: {}'.format(op))

                    do_apply = jax.random.bernoulli(key, self.probability)
                    x = jnp.where(do_apply, x_new, x)

                if invert:
                    warnings.warn("Trying to invert a normalized color jittered image, which is not invertible.")
                current = x
            else:
                current = input
            val.append(current)
        return val