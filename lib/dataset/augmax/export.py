import jax

from .geometric import RandomSizedCrop, Rotate, HorizontalFlip, RandomTranslate
from .imagelevel import NormalizedColorJitter, Cutout


def get_vmap_transform(transform, use_siamese=False):
    if use_siamese:
        vmap_transform = jax.vmap(transform, in_axes=[None, 0])
    else:
        transform = jax.vmap(transform, in_axes=[0, 0])

        def vmap_transform(rng, img):
            bs = img.shape[0]
            rngs = jax.random.split(rng, bs)
            return transform(rngs, img)

    return vmap_transform


def get_aug_by_name(strategy, res=32):
    transform = dict(color=jax.jit(get_vmap_transform(NormalizedColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.25, p=1.0), use_siamese=False)),
        crop=jax.jit(
            get_vmap_transform(RandomSizedCrop(width=res, height=res, zoom_range=(0.8, 1.25)), use_siamese=False)),
        translate=jax.jit(get_vmap_transform(RandomTranslate(ratio=0.125), use_siamese=False)),
        cutout=jax.jit(
            get_vmap_transform(Cutout(num_holes=1, max_h_size=res // 4, max_w_size=res // 4, fill_value=0.0, p=1.0),
                               use_siamese=False)),
        flip=jax.jit(get_vmap_transform(HorizontalFlip(p=0.5), use_siamese=False)),
        rotate=jax.jit(get_vmap_transform(Rotate(angle_range=(-15, 15), p=1.0), use_siamese=False)))

    strategy = strategy.split('_')

    def trans(rng, x):
        i = jax.random.randint(key=rng, shape=(1,), minval=0, maxval=len(strategy))[0]
        return transform[strategy[i]](rng, x)

    return trans
