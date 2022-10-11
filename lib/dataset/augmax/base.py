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
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Sequence
from enum import Enum

from .utils import unpack_list_if_singleton

 
class InputType(Enum):
    IMAGE = 'image'
    MASK = 'mask'
    DENSE = 'dense'
    CONTOUR = 'contour'
    KEYPOINTS = 'keypoints'


def same_type(left_type, right_type):
    if isinstance(left_type, InputType):
        left_type = left_type.value
    if isinstance(right_type, InputType):
        right_type = right_type.value
    return left_type.lower() == right_type.lower()


class Transformation(ABC):
    def __init__(self, input_types=None):
            if input_types is None:
                self.input_types = [InputType.IMAGE]
            else:
                self.input_types = input_types

    def __call__(self, rng: jnp.ndarray, *inputs: jnp.ndarray) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
        if len(self.input_types) != len(inputs):
            raise ValueError(f"List of input types (length {len(self.input_types)}) must match inputs to Augmentation (length {len(inputs)})")
        augmented = self.apply(rng, inputs, self.input_types)
        return unpack_list_if_singleton(augmented)

    def invert(self, rng: jnp.ndarray, *inputs: jnp.ndarray) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
        if len(self.input_types) != len(inputs):
            raise ValueError(f"List of input types (length {len(self.input_types)}) must match inputs to Augmentation (length {len(inputs)})")
        augmented = self.apply(rng, inputs, self.input_types, invert=True)
        return unpack_list_if_singleton(augmented)

    @abstractmethod
    def apply(self, rng: jnp.ndarray, inputs: Sequence[jnp.ndarray], input_types: Sequence[InputType]=None, invert=False) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types
        val = []
        for input, type in zip(inputs, input_types):
            val.append(input)
        return val


class BaseChain(Transformation):
    def __init__(self, *transforms: Transformation, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        self.transforms = transforms

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: Sequence[InputType]=None, invert=False) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        N = len(self.transforms)
        subkeys = [None]*N if rng is None else jax.random.split(rng, N)

        transforms = self.transforms
        if invert:
            transforms = reversed(transforms)
            subkeys = reversed(subkeys)

        images = list(inputs)
        for transform, subkey in zip(transforms, subkeys):
            images = transform.apply(subkey, images, input_types, invert=invert)
        return images 

    def __repr__(self):
        members_repr = ",\n".join(str(t) for t in self.transforms)
        members_repr = '\n'.join(['\t'+line for line in members_repr.split('\n')])
        return f'{self.__class__.__name__}(\n{members_repr}\n)'
