import sys

sys.path.append("..")

import os
import fire
import optax
import ml_collections
from functools import partial
from datetime import datetime

import jax
from absl import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from flax.training import checkpoints

from lib.dataset.dataloader import get_dataset, configure_dataloader
from lib.models.utils import create_model
from lib.datadistillation.frepo import proto_evaluate
from lib.training.utils import create_train_state
from lib.dataset.augmax import get_aug_by_name

from clu import metric_writers


def get_config():
    config = ml_collections.ConfigDict()
    config.random_seed = 0
    config.train_log = 'train_log'
    config.train_img = 'train_img'
    config.resume = True

    config.img_size = None
    config.img_channels = None
    config.num_prototypes = None
    config.train_size = None

    config.dataset = ml_collections.ConfigDict()
    config.kernel = ml_collections.ConfigDict()
    config.online = ml_collections.ConfigDict()

    # Dataset
    config.dataset.name = 'cifar100'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'tiny_imagenet']
    config.dataset.data_path = 'data/tensorflow_datasets'
    config.dataset.zca_path = 'data/zca'
    config.dataset.zca_reg = 0.1

    # online
    config.online.img_size = None
    config.online.img_channels = None
    config.online.optimizer = 'adam'
    config.online.learning_rate = 0.0003
    config.online.arch = 'conv'
    config.online.output = 'feat_fc'
    config.online.width = 128
    config.online.normalization = 'identity'

    # Kernel
    config.kernel.img_size = None
    config.kernel.img_channels = None
    config.kernel.num_prototypes = None
    config.kernel.train_size = None
    config.kernel.resume = config.resume
    config.kernel.optimizer = 'lamb'
    config.kernel.learning_rate = 0.0003
    config.kernel.batch_size = 1024
    config.kernel.eval_batch_size = 1000

    return config


def get_chunk_ds(x_train, y_train, x_test, y_test, chunk_size, chunk_idx):
    start_idx = chunk_size * chunk_idx

    x_train = x_train[start_idx:start_idx + chunk_size]
    y_train = y_train[start_idx:start_idx + chunk_size]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return ds_train, ds_test


def load_proto(ckpt_dir, ckpt_name, use_chunk=False, chunk_size=2000, use_cl=False, steps=0, idx=0,
               seed=0):
    def load_ckpt(ckpt_path, prefix='checkpoint_'):
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint path {} does not exists!'.format(ckpt_path))
        state = checkpoints.restore_checkpoint(ckpt_path, None, prefix=prefix)
        x_proto = state['params']['x_proto']
        y_proto = state['params']['y_proto']
        return x_proto, y_proto

    if use_chunk:
        x_proto = []
        y_proto = []
        for idx in range(len(os.listdir(ckpt_dir))):
            ckpt_path = os.path.join(ckpt_dir, 'csize{}_cidx{}'.format(chunk_size, idx), ckpt_name)
            x, y = load_ckpt(ckpt_path)
            x_proto.append(x)
            y_proto.append(y)
        x_proto = np.concatenate(x_proto, axis=0)
        y_proto = np.concatenate(y_proto, axis=0)
    elif use_cl:
        x_proto = []
        y_proto = []
        for i in range(idx + 1):
            ckpt_path = os.path.join(ckpt_dir, 'steps{}_idx{}_seed{}'.format(steps, i, seed), ckpt_name)
            x, y = load_ckpt(ckpt_path)
            x_proto.append(x)
            y_proto.append(y)
        x_proto = np.concatenate(x_proto, axis=0)
        y_proto = np.concatenate(y_proto, axis=0)
    else:
        if ckpt_name != '':
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            x_proto, y_proto = load_ckpt(ckpt_path)
        else:
            ckpt_path = ckpt_dir
            x_proto, y_proto = load_ckpt(ckpt_path, prefix='')
    return x_proto, y_proto


def main(dataset_name, data_path=None, zca_path=None, ckpt_dir=None, ckpt_name='', res_dir=None, random_seed=0,
         eval_batch_size=1000, arch='conv', width=128, depth=3, normalization='identity', pooling='avg',
         use_chunk=False, chunk_size=2000, optimizer='adam', learning_rate=0.0003, weight_decay=0.0003,
         loss='mse', temperature=1.0, num_eval=5):
    # --------------------------------------
    # Setup
    # --------------------------------------
    config = get_config()
    config.random_seed = random_seed

    use_pmap = jax.device_count('gpu') > 1
    if use_pmap:
        logging.info('Use Multi GPU Training. \n Number of GPUs: {}'.format(jax.device_count()))

    # --------------------------------------
    # Dataset
    # --------------------------------------
    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset)

    # --------------------------------------
    # Online
    # --------------------------------------
    config.online.arch = arch
    config.online.width = width
    config.online.depth = depth
    config.online.normalization = normalization
    config.online.img_size = config.dataset.img_shape[0]
    config.online.img_channels = config.dataset.img_shape[-1]
    config.online.optimizer = optimizer
    config.online.weight_decay = weight_decay

    # --------------------------------------
    # Logging
    # --------------------------------------
    if dataset_name in ['mnist', 'fashion_mnist']:
        use_flip = False
        aug_strategy = 'color_crop_rotate_translate_cutout'
    else:
        use_flip = True
        aug_strategy = 'flip_color_crop_rotate_translate_cutout'

    x_proto, y_proto = load_proto(ckpt_dir, ckpt_name, use_chunk=use_chunk, chunk_size=chunk_size)
    num_prototypes = x_proto.shape[0]

    logging.info('x_proto shape {}, y_proto shape {}'.format(x_proto.shape, y_proto.shape))

    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    if 'train_log' in ckpt_dir:
        workdir = ckpt_dir.replace('train_log', 'eval_log')
    else:
        workdir = os.path.join('train_log', ckpt_dir)

    hparam = '{}_w{}_{}/{}_lr{}_wd{}_aug{}'.format(arch, width, normalization, optimizer, learning_rate,
                                                   weight_decay, aug_strategy)
    workdir = os.path.join(workdir, hparam)
    writer = metric_writers.create_default_writer(logdir=workdir)
    logging.info('Working directory: {}'.format(workdir))

    # --------------------------------------
    # Creat model
    # --------------------------------------
    # Centered one hot label for mse loss
    y_transform = lambda y: tf.one_hot(y, config.dataset.num_classes, on_value=1 - 1 / config.dataset.num_classes,
                                       off_value=-1 / config.dataset.num_classes)

    if loss == 'softxent':
        y_proto = y_proto / temperature

    ds_test = configure_dataloader(ds_test, batch_size=eval_batch_size, y_transform=y_transform, train=False,
                                   shuffle=False)

    if normalization == 'batch':
        normalization = 'identity'

    has_bn = False

    # Rough estimate of number of steps to train on distilled data
    step_per_prototpyes = {10: 1000, 100: 2000, 200: 20000, 400: 5000, 500: 5000, 1000: 10000, 2000: 40000, 5000: 40000}
    num_online_eval_updates = step_per_prototpyes[num_prototypes]
    steps_per_log = num_online_eval_updates / 10

    logging.info(
        '\n=========== Num_online_updates: {}, Steps_per_log:{} '.format(num_online_eval_updates, steps_per_log))

    logging.info(
        '\n=========== {} ({}) has_bn: {} lr={} ============\n'.format(arch, normalization, has_bn, learning_rate))

    eval_model = create_model(arch, config.dataset.num_classes, width=width, depth=depth,
                              normalization=normalization, pooling=pooling, output=config.online.output)

    warmup_steps = 500
    warmup_fn = optax.linear_schedule(init_value=0., end_value=learning_rate, transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(init_value=learning_rate, alpha=0.01,
                                            decay_steps=max(num_online_eval_updates - warmup_steps, 1))
    learning_rate_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])

    create_eval_state = partial(create_train_state, config=config.online, model=eval_model,
                                learning_rate_fn=learning_rate_fn, has_bn=has_bn)

    ds_proto = tf.data.Dataset.from_tensor_slices((x_proto, y_proto))
    ds_proto = ds_proto.cache().repeat().batch(batch_size=min(y_proto.shape[0], 500))

    _, acc = proto_evaluate(ds_proto, ds_test, workdir, create_eval_state, seed=random_seed, use_flip=use_flip,
                            loss=loss, num_online_eval_updates=num_online_eval_updates, diff_aug=diff_aug,
                            steps_per_log=steps_per_log, writer=writer, has_bn=has_bn, use_pmap=use_pmap,
                            x_proto=x_proto, y_proto=y_proto, num_eval=num_eval)

    df = pd.DataFrame([[dataset_name, ckpt_dir, ckpt_name, arch, width, depth, np.mean(acc), np.std(acc),
                        datetime.now().strftime("%y%m%d %H:%M:%S")]],
                      columns=['dataset', 'ckpt_dir', 'ckpt_name', 'arch', 'width', 'depth', 'mean', 'std',
                               'timestamp'])

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    res_file = f'{res_dir}/summary.csv'

    if os.path.exists(res_file):
        old_df = pd.read_csv(res_file)
        df = old_df.append(df)
    df.to_csv(res_file, sep=',', encoding='utf-8', index=False)

    logging.info('Finish!')


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
