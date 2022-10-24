import flax.linen as nn
import functools
from typing import (Callable)


class DC_ConvNet(nn.Module):
    depth: int = 3
    width: int = 128
    kernel_size: tuple = (3, 3)
    activation_fn: Callable = nn.relu
    use_gap: bool = False
    num_classes: int = 10
    kernel_init: functools.partial = nn.initializers.kaiming_normal()
    bias_init: functools.partial = nn.initializers.zeros
    normalization: str = 'identity'
    pooling: str = 'avg'
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        if self.output not in ['softmax', 'log_softmax', 'logit', 'feat_fc']:
            raise ValueError(
                'Wrong argument. Possible choices for output are "softmax", "log_softmax", "logit",and "feat_fc".')

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm,
                                           use_running_average=not train,
                                           epsilon=1e-05,
                                           momentum=0.9,
                                           dtype=self.dtype)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = None
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        if self.pooling == 'avg':
            pool_layer = nn.avg_pool
        elif self.pooling == 'max':
            pool_layer = nn.max_pool
        elif self.pooling == 'identity':
            pool_layer = lambda x, *args, **kargs: x
        else:
            raise ValueError('Unknown Pooling Layer {}!'.format(self.pooling))

        # generate blocks of convolutions followed by average pooling (n, 32, 32, 512)
        for i in range(self.depth):
            x = nn.Conv(features=self.width, kernel_size=self.kernel_size, kernel_init=self.kernel_init, use_bias=False,
                        dtype=self.dtype)(x)

            if not self.normalization == 'identity':
                x = norm_layer()(x)

            x = self.activation_fn(x)
            x = pool_layer(x, (2, 2), strides=(2, 2))

        if self.use_gap:
            x = nn.avg_pool(x, x.shape[1:3])
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.reshape((x.shape[0], -1))

        feat_fc = x

        x = nn.Dense(features=self.num_classes,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                     dtype=self.dtype)(x)

        if self.output == 'logit':
            return x
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'feat_fc':
            return x, feat_fc


class KIP_ConvNet(nn.Module):
    depth: int = 3
    width: int = 128
    kernel_size: tuple = (3, 3)
    activation_fn: Callable = nn.relu
    use_gap: bool = False
    num_classes: int = 10
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = functools.partial(nn.initializers.normal, stddev=0.1)()
    normalization: str = 'identity'
    pooling: str = 'avg'
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        if self.output not in ['softmax', 'log_softmax', 'logit', 'activations', 'feat_conv', 'feat_fc']:
            raise ValueError(
                'Wrong argument. Possible choices for output are "softmax", "log_softmax", "logit", "activations", "feat_conv", and "feat_fc".')

        act = {}

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm,
                                           use_running_average=not train,
                                           epsilon=1e-05,
                                           momentum=0.1,
                                           dtype=self.dtype)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = None
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        if self.pooling == 'avg':
            pool_layer = nn.avg_pool
        elif self.pooling == 'max':
            pool_layer = nn.max_pool
        elif self.pooling == 'identity':
            pool_layer = lambda x, *args, **kargs: x
        else:
            raise ValueError('Unknown Pooling Layer {}!'.format(self.pooling))

        x = nn.Conv(features=self.width, kernel_size=self.kernel_size, kernel_init=self.kernel_init, use_bias=True,
                    dtype=self.dtype)(x)

        act['conv0'] = x
        x = self.activation_fn(x)

        # generate blocks of convolutions followed by average pooling (n, 32, 32, 512)
        for i in range(self.depth):
            if not self.normalization == 'identity':
                x = norm_layer()(x)

            x = nn.Conv(features=self.width, kernel_size=self.kernel_size, kernel_init=self.kernel_init, use_bias=True,
                        dtype=self.dtype)(x)
            act['conv{}'.format(i + 1)] = x

            x = self.activation_fn(x)
            x = pool_layer(x, (2, 2), strides=(2, 2))

        feat_conv = x  # (n, 4, 4, 512)
        if self.use_gap:
            x = nn.avg_pool(x, x.shape[1:3])
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.reshape((x.shape[0], -1))

        feat_fc = x

        x = nn.Dense(features=self.num_classes,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                     dtype=self.dtype)(x)
        act['fc'] = x  # (n, 512)

        if self.output == 'logit':
            return x
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'activations':
            return act
        if self.output == 'feat_conv':
            return x, feat_conv
        if self.output == 'feat_fc':
            return x, feat_fc


class Conv(nn.Module):
    depth: int = 3
    width: int = 128
    kernel_size: tuple = (3, 3)
    num_classes: int = 10
    normalization: str = 'identity'
    pooling: str = 'avg'
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        channel = x.shape[-1]

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm, use_running_average=not train, epsilon=1e-05, momentum=0.1,
                                           dtype=self.dtype)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = None
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        for i in range(self.depth):
            if i != 0 and self.normalization != 'identity':
                x = norm_layer()(x)

            if i == 0 and channel == 1:
                pad = (self.kernel_size[0] // 2 + 2, self.kernel_size[0] // 2 + 2)
            else:
                pad = (self.kernel_size[0] // 2, self.kernel_size[0] // 2)

            x = nn.Conv(features=self.width * (2 ** i), kernel_size=self.kernel_size,
                        padding=(pad, pad), use_bias=True, dtype=self.dtype)(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        feat_fc = x

        x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)

        if self.output == 'logit':
            return x
        if self.output == 'feat_fc':
            return x, feat_fc


''' AlexNet '''


class AlexNet(nn.Module):
    num_classes: int = 10
    pooling: str = 'max'
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        channel = x.shape[-1]

        if self.output not in ['softmax', 'log_softmax', 'logit', 'feat_fc']:
            raise ValueError(
                'Wrong argument. Possible choices for output are "softmax", "log_softmax", "logit",and "feat_fc".')

        if self.pooling == 'avg':
            pool_layer = nn.avg_pool
        elif self.pooling == 'max':
            pool_layer = nn.max_pool
        elif self.pooling == 'identity':
            pool_layer = lambda x, *args, **kargs: x
        else:
            raise ValueError('Unknown Pooling Layer {}!'.format(self.pooling))

        if channel == 1:
            pad = (5 // 2 + 2, 5 // 2 + 2)
        else:
            pad = (5 // 2, 5 // 2)

        x = nn.Conv(features=128, kernel_size=(5, 5), padding=(pad, pad))(x)
        x = pool_layer(nn.relu(x), (2, 2), strides=(2, 2))
        x = nn.Conv(features=192, kernel_size=(5, 5), padding='SAME')(x)
        x = pool_layer(nn.relu(x), (2, 2), strides=(2, 2))
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=192, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=192, kernel_size=(3, 3), padding='SAME')(x)
        x = pool_layer(nn.relu(x), (2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        feat_fc = x

        x = nn.Dense(features=self.num_classes)(x)

        if self.output == 'logit':
            return x
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'feat_fc':
            return x, feat_fc


''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    architecture: str = 'VGG11'
    num_classes: int = 10
    pooling: str = 'max'
    normalization: str = 'identity'
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        channel = x.shape[-1]
        cfg = cfg_vgg[self.architecture]

        if self.output not in ['softmax', 'log_softmax', 'logit', 'feat_fc']:
            raise ValueError(
                'Wrong argument. Possible choices for output are "softmax", "log_softmax", "logit",and "feat_fc".')

        if self.pooling == 'avg':
            pool_layer = nn.avg_pool
        elif self.pooling == 'max':
            pool_layer = nn.max_pool
        elif self.pooling == 'identity':
            pool_layer = lambda x, *args, **kargs: x
        else:
            raise ValueError('Unknown Pooling Layer {}!'.format(self.pooling))

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm, use_running_average=not train, momentum=0.9)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = lambda: lambda x: x
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        if channel == 1:
            pad = (3 // 2 + 2, 3 // 2 + 2)
        else:
            pad = (3 // 2, 3 // 2)

        for ic, w in enumerate(cfg):
            if w == 'M':
                x = pool_layer(x, (2, 2), strides=(2, 2))
            else:
                if ic == 0:
                    x = nn.Conv(features=128, kernel_size=(3, 3), padding=(pad, pad))(x)
                else:
                    x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
                x = norm_layer()(x)
                x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        feat_fc = x

        x = nn.Dense(features=self.num_classes)(x)

        if self.output == 'logit':
            return x
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'feat_fc':
            return x, feat_fc


def VGG11(num_classes, pooling, normalization, output):
    return VGG('VGG11', num_classes, pooling, normalization, output)


def VGG13(num_classes, pooling, normalization, output):
    return VGG('VGG13', num_classes, pooling, normalization, output)


def VGG16(num_classes, pooling, normalization, output):
    return VGG('VGG16', num_classes, pooling, normalization, output)


def VGG19(num_classes, pooling, normalization, output):
    return VGG('VGG19', num_classes, pooling, normalization, output)


''' ResNet '''

LAYERS = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 8, 36, 3],
          'resnet20': [3, 3, 3],
          'resnet32': [5, 5, 5],
          'resnet44': [7, 7, 7],
          'resnet56': [9, 9, 9],
          'resnet110': [18, 18, 18],
          }


class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    kernel_size: tuple = (3, 3)
    normalization: str = 'identity'
    block_name: str = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm, use_running_average=not train, momentum=0.9)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = lambda: lambda x: x
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        residual = x

        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=(self.stride, self.stride),
                    padding=((1, 1), (1, 1)), use_bias=False, dtype=self.dtype)(x)
        x = norm_layer()(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=(1, 1),
                    padding=((1, 1), (1, 1)), use_bias=False, dtype=self.dtype)(x)
        x = norm_layer()(x)

        if self.stride != 1 or (x.shape[-1] != residual.shape[-1]):
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=(self.stride, self.stride),
                               use_bias=False, dtype=self.dtype)(residual)
            residual = norm_layer()(residual)

        x += residual
        x = nn.relu(x)
        return x


class BasicBlock_AP(nn.Module):
    features: int
    stride: int = 1
    kernel_size: tuple = (3, 3)
    pooling: str = 'max'
    normalization: str = 'identity'
    block_name: str = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        if self.pooling == 'avg':
            pool_layer = nn.avg_pool
        elif self.pooling == 'max':
            pool_layer = nn.max_pool
        elif self.pooling == 'identity':
            pool_layer = lambda x, *args, **kargs: x
        else:
            raise ValueError('Unknown Pooling Layer {}!'.format(self.pooling))

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm, use_running_average=not train, momentum=0.9)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = lambda: lambda x: x
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        residual = x

        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=(1, 1),
                    padding=((1, 1), (1, 1)), use_bias=False, dtype=self.dtype)(x)

        x = norm_layer()(x)
        x = nn.relu(x)

        if self.stride != 1:
            x = pool_layer(x, (2, 2), strides=(2, 2))

        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=(1, 1),
                    padding=((1, 1), (1, 1)), use_bias=False, dtype=self.dtype)(x)
        x = norm_layer()(x)

        if self.stride != 1 or (x.shape[-1] != residual.shape[-1]):
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=(1, 1),
                               use_bias=False, dtype=self.dtype)(residual)
            if self.stride != 1:
                residual = pool_layer(residual, (2, 2), strides=(2, 2))
            residual = norm_layer()(residual)

        x += residual
        x = nn.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logit': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str):
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    architecture: str = 'resnet18'
    num_classes: int = 10
    normalization: str = 'identity'
    block: nn.Module = BasicBlock
    output: str = 'softmax'
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logit' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """

        if self.normalization == 'batch':
            norm_layer = functools.partial(nn.BatchNorm, use_running_average=not train, momentum=0.9)
        elif self.normalization == 'layer':
            norm_layer = functools.partial(nn.LayerNorm, dtype=self.dtype)
        elif self.normalization == 'group':
            norm_layer = functools.partial(nn.GroupNorm, dtype=self.dtype)
        elif self.normalization == 'group1':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=1, dtype=self.dtype)
        elif self.normalization == 'instance':
            norm_layer = functools.partial(nn.GroupNorm, num_groups=None, group_size=1, dtype=self.dtype)
        elif self.normalization == 'identity':
            norm_layer = lambda: lambda x: x
        else:
            raise ValueError('Unknown Normalization Layer {}!'.format(self.normalization))

        x = nn.Conv(features=64, kernel_size=(3, 3), use_bias=False, dtype=self.dtype)(x)

        x = norm_layer()(x)
        x = nn.relu(x)

        for i in range(LAYERS[self.architecture][0]):
            x = self.block(features=64, kernel_size=(3, 3), stride=1,
                           block_name=f'block1_{i}', dtype=self.dtype)(x, train)

        for i in range(LAYERS[self.architecture][1]):
            x = self.block(features=128, kernel_size=(3, 3), stride=2 if i == 0 else 1,
                           block_name=f'block1_{i}', dtype=self.dtype)(x, train)

        for i in range(LAYERS[self.architecture][2]):
            x = self.block(features=256, kernel_size=(3, 3), stride=2 if i == 0 else 1,
                           block_name=f'block2_{i}', dtype=self.dtype)(x, train)

        for i in range(LAYERS[self.architecture][3]):
            x = self.block(features=512, kernel_size=(3, 3), stride=2 if i == 0 else 1,
                           block_name=f'block3_{i}', dtype=self.dtype)(x, train)

        # Classifier
        x = x.reshape((x.shape[0], -1))

        feat_fc = x  # (n, 64)

        x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)

        if self.output == 'logit':
            return x
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'feat_fc':
            return x, feat_fc


def ResNet18(num_classes, pooling, normalization, output):
    return ResNet('resnet18', num_classes, normalization, BasicBlock, output)


def ResNet18_AP(num_classes, pooling, normalization, output):
    return ResNet('resnet18', num_classes, normalization, BasicBlock_AP, output)
