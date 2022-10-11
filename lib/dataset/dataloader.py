from absl import logging

import os
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from .imagewoof import ImagewoofV2
from .imagenette import ImagenetteV2
from .tinyimagenet import TinyImagenetV2
from .ops import compute_zca_mean_cov_ds, get_whitening_transform, get_preprocess_op_np, load_data, process2tfrecord

# Precomputed mean and std
data_stats = {
    'mnist': ([0.1307], [0.3081]),
    'fashion_mnist': ([0.2861], [0.3530]),
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    'cifar100': ([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
    'tiny_imagenet': ([0.4759, 0.4481, 0.3926], [0.2763, 0.2687, 0.2813]),
    'imagenette': ([0.4626, 0.4588, 0.4251], [0.2790, 0.2745, 0.2973]),
    'imagewoof': ([0.4917, 0.4613, 0.3931], [0.2513, 0.2442, 0.2530]),
    'imagenet_resized/32x32': ([0.4811, 0.4575, 0.4079], [0.2604, 0.2532, 0.2682]),
    'imagenet_resized/64x64': ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
    'caltech_birds2011': ([0.4810, 0.4964, 0.4245], [0.2129, 0.2084, 0.2468])
}


def get_ds_builder(dataset_name, data_dir):
    if dataset_name == 'imagewoof':
        ds_builder = ImagewoofV2(data_dir=data_dir)
    elif dataset_name == 'imagenette':
        ds_builder = ImagenetteV2(data_dir=data_dir)
    elif dataset_name == 'tiny_imagenet':
        ds_builder = TinyImagenetV2(data_dir=data_dir)
    else:
        ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds_builder.download_and_prepare()
    return ds_builder


def configure_dataloader(ds, batch_size, x_transform=None, y_transform=None, train=False, shuffle=False, seed=0):
    if y_transform is None:
        y_transform = lambda x: x
    else:
        y_transform = y_transform

    ds = ds.cache()
    if train:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(16 * batch_size, seed=seed)

    if x_transform:
        ds = ds.map(lambda x, y: (x_transform(x), y_transform(y)), tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (x, y_transform(y)), tf.data.AUTOTUNE)

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_dataset(config, return_raw=False):
    dataset_name = config.name
    data_path = config.data_path
    zca_path = config.zca_path
    zca_reg = config.zca_reg

    if dataset_name in ['imagenet_resized/64x64', 'imagenette', 'imagewoof']:
        split = ['train', 'validation']
    else:
        split = ['train', 'test']

    if dataset_name in ['mnist', 'fashion_mnist']:
        preprocess_type = 'standard'
    else:
        preprocess_type = 'normalize_zca'

    if dataset_name in ['imagenette', 'imagewoof']:
        use_checkboard = True
        use_mean_block = True
        block_size = 64
        resolution = 128
    elif dataset_name in ['imagenet_resized/64x64', 'tiny_imagenet']:
        use_checkboard = False
        use_mean_block = False
        block_size = None
        resolution = 64
    else:
        use_checkboard = False
        use_mean_block = False
        block_size = None
        resolution = 32

    ds_builder = get_ds_builder(dataset_name, data_path)
    img_shape = ds_builder.info.features['image'].shape
    num_train, num_test = ds_builder.info.splits[split[0]].num_examples, ds_builder.info.splits[split[1]].num_examples
    num_classes, class_names = ds_builder.info.features['label'].num_classes, ds_builder.info.features['label'].names

    mean, std = data_stats[dataset_name]
    mean, std = np.array(mean), np.array(std)

    if preprocess_type == 'standard':
        zca_mean, whitening_transform, rev_whitening_transform = None, None, None
    elif preprocess_type == 'normalize_zca':
        if not os.path.exists(zca_path):
            os.makedirs(zca_path)

        if '/' in dataset_name:
            name = dataset_name.split('/')[0]
            if not os.path.exists('{}/{}'.format(zca_path, name)):
                os.makedirs('{}/{}'.format(zca_path, name))

        if block_size is None:
            path = os.path.join(zca_path, '{}_{}.npz'.format(dataset_name, preprocess_type))
        else:
            if use_checkboard:
                path = os.path.join(zca_path,
                                    '{}_{}_res{}_block{}_mean{}_cb.npz'.format(dataset_name, preprocess_type,
                                                                               resolution, block_size, use_mean_block))
            else:
                path = os.path.join(zca_path,
                                    '{}_{}_res{}_block{}_mean{}.npz'.format(dataset_name, preprocess_type,
                                                                            resolution, block_size, use_mean_block))

        if not os.path.exists(path):
            logging.info('Compute block zca with block_size {} and save to {}!'.format(block_size, path))
            ds_train = ds_builder.as_dataset(split='train', as_supervised=True)

            zca_mean, cov = compute_zca_mean_cov_ds(ds_train, img_shape, mean=mean, std=std, resolution=resolution,
                                                    block_size=block_size, batch_size=10000,
                                                    use_checkboard=use_checkboard)
            whitening_transform, rev_whitening_transform = get_whitening_transform(cov, num_train, zca_reg=zca_reg,
                                                                                   use_mean_block=use_mean_block)
            np.savez(path, whitening_transform=whitening_transform, rev_whitening_transform=rev_whitening_transform,
                     zca_mean=zca_mean)
        else:
            logging.info('Load from {}!'.format(path))
            npzfile = np.load(path)
            whitening_transform, rev_whitening_transform, zca_mean = npzfile['whitening_transform'], npzfile[
                'rev_whitening_transform'], npzfile['zca_mean']
    else:
        raise ValueError('Unknown PreprocessType {}!'.format(preprocess_type))

    preprocess_op, rev_preprocess_op = get_preprocess_op_np(mean=mean, std=std, zca_mean=zca_mean,
                                                            whitening_transform=whitening_transform,
                                                            rev_whitening_transform=rev_whitening_transform,
                                                            block_size=block_size, use_mean_block=use_mean_block,
                                                            use_checkboard=use_checkboard)

    ds_train, ds_test = ds_builder.as_dataset(split=split, as_supervised=True)

    if dataset_name in ['imagenet_resized/64x64', 'caltech_birds2011']:
        data_dir = os.path.join(zca_path,
                                '{}_{}_res{}_block{}_mean{}'.format(dataset_name, preprocess_type, resolution,
                                                                    block_size, use_mean_block))

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

            if '/' in dataset_name:
                name = dataset_name.split('/')[0]
            else:
                name = dataset_name
            process2tfrecord(ds_train, ds_test, data_dir, name, img_shape, num_classes, preprocess_op,
                             resolution, batch_size=10000, num_per_shard=10000)

        builder = tfds.core.builder_from_directory(data_dir)
        ds_train, ds_test = builder.as_dataset(split=['train', 'test'], as_supervised=True, shuffle_files=True)
        x_train, y_train, x_test, y_test = None, None, None, None

    else:
        x_train, y_train = load_data(ds_train, img_shape, preprocess_op, resolution, batch_size=5000)
        x_test, y_test = load_data(ds_test, img_shape, preprocess_op, resolution, batch_size=5000)
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    proto_scale = {'x_proto': jnp.sqrt(3 * (resolution ** 2))}

    logging.info('Resolution: {}'.format(resolution))
    logging.info('Proto Scale: {}'.format(proto_scale))

    with config.unlocked():
        config.img_shape = (resolution, resolution, 3) if None in img_shape else img_shape
        config.num_classes = num_classes
        config.class_names = class_names
        config.train_size = num_train
        config.test_size = num_test

    if return_raw:
        return (x_train, y_train, x_test, y_test), preprocess_op, rev_preprocess_op, proto_scale
    else:
        return (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale
