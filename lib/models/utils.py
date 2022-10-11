import jax.numpy as jnp
import flax.linen as nn
from .networks import Conv, DC_ConvNet, KIP_ConvNet, AlexNet, VGG11, ResNet18, ResNet18_AP


def create_model(arch, num_classes, normalization='identity', output='logit',
                 width=512, depth=3, pooling='avg', use_gap=False, kernel_size=(3, 3), activation_fn=nn.relu,
                 dtype=jnp.float32, **kwargs):
    if arch == 'resnet18':
        model = ResNet18(output=output, num_classes=num_classes, pooling=pooling, normalization=normalization)
    elif arch == 'resnet18_ap':
        model = ResNet18_AP(output=output, num_classes=num_classes, pooling=pooling, normalization=normalization)
    elif arch == 'alexnet':
        model = AlexNet(output=output, num_classes=num_classes, pooling=pooling, **kwargs)
    elif arch == 'vgg11':
        model = VGG11(output=output, num_classes=num_classes, pooling=pooling, normalization=normalization)
    elif arch == 'dcconv':
        model = DC_ConvNet(output=output, num_classes=num_classes, normalization=normalization, width=width,
                           depth=depth, kernel_size=kernel_size, activation_fn=activation_fn, use_gap=use_gap,
                           pooling=pooling,
                           dtype=dtype, **kwargs)
    elif arch == 'kipconv':
        model = KIP_ConvNet(output=output, num_classes=num_classes, normalization=normalization, width=width,
                            depth=depth, kernel_size=kernel_size, activation_fn=activation_fn, use_gap=use_gap,
                            pooling=pooling, dtype=dtype, **kwargs)
    elif arch == 'conv':
        model = Conv(output=output, num_classes=num_classes, normalization=normalization, width=width, depth=depth,
                     kernel_size=kernel_size, pooling=pooling, dtype=dtype, **kwargs)
    else:
        raise ValueError('Unknown Architecture: {}!'.format(arch))

    return model
