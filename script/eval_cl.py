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

class_order = dict(
    seed0=[26, 86, 2, 55, 75, 93, 16, 73, 54, 95, 53, 92, 78, 13, 7, 30, 22, 24, 33, 8, 43, 62, 3, 71, 45, 48, 6, 99,
           82, 76, 60, 80, 90, 68, 51, 27, 18, 56, 63, 74, 1, 61, 42, 41, 4, 15, 17, 40, 38, 5, 91, 59, 0, 34, 28, 50,
           11, 35, 23, 52, 10, 31, 66, 57, 79, 85, 32, 84, 14, 89, 19, 29, 49, 97, 98, 69, 20, 94, 72, 77, 25, 37, 81,
           46, 39, 65, 58, 12, 88, 70, 87, 36, 21, 83, 9, 96, 67, 64, 47, 44],
    seed1=[80, 84, 33, 81, 93, 17, 36, 82, 69, 65, 92, 39, 56, 52, 51, 32, 31, 44, 78, 10, 2, 73, 97, 62, 19, 35, 94,
           27, 46, 38, 67, 99, 54, 95, 88, 40, 48, 59, 23, 34, 86, 53, 77, 15, 83, 41, 45, 91, 26, 98, 43, 55, 24, 4,
           58, 49, 21, 87, 3, 74, 30, 66, 70, 42, 47, 89, 8, 60, 0, 90, 57, 22, 61, 63, 7, 96, 13, 68, 85, 14, 29, 28,
           11, 18, 20, 50, 25, 6, 71, 76, 1, 16, 64, 79, 5, 75, 9, 72, 12, 37],
    seed2=[83, 30, 56, 24, 16, 23, 2, 27, 28, 13, 99, 92, 76, 14, 0, 21, 3, 29, 61, 79, 35, 11, 84, 44, 73, 5, 25, 77,
           74, 62, 65, 1, 18, 48, 36, 78, 6, 89, 91, 10, 12, 53, 87, 54, 95, 32, 19, 26, 60, 55, 9, 96, 17, 59, 57, 41,
           64, 45, 97, 8, 71, 94, 90, 98, 86, 80, 50, 52, 66, 88, 70, 46, 68, 69, 81, 58, 33, 38, 51, 42, 4, 67, 39, 37,
           20, 31, 63, 47, 85, 93, 49, 34, 7, 75, 82, 43, 22, 72, 15, 40],
    seed3=[93, 67, 6, 64, 96, 83, 98, 42, 25, 15, 77, 9, 71, 97, 34, 75, 82, 23, 59, 45, 73, 12, 8, 4, 79, 86, 17, 65,
           47, 50, 30, 5, 13, 31, 88, 11, 58, 85, 32, 40, 16, 27, 35, 36, 92, 90, 78, 76, 68, 46, 53, 70, 80, 61, 18,
           91, 57, 95, 54, 55, 28, 52, 84, 89, 49, 87, 37, 48, 33, 43, 7, 62, 99, 29, 69, 51, 1, 60, 63, 2, 66, 22, 81,
           26, 14, 39, 44, 20, 38, 94, 10, 41, 74, 19, 21, 0, 72, 56, 3, 24],
    seed4=[20, 10, 96, 16, 63, 24, 53, 97, 41, 47, 43, 2, 95, 26, 13, 37, 14, 29, 35, 54, 80, 4, 81, 76, 85, 60, 5, 70,
           71, 19, 65, 62, 27, 75, 61, 78, 18, 88, 7, 39, 6, 77, 11, 59, 22, 94, 23, 12, 92, 25, 83, 48, 17, 68, 31, 34,
           15, 51, 86, 82, 28, 64, 67, 33, 45, 42, 40, 32, 91, 74, 49, 8, 30, 99, 66, 56, 84, 73, 79, 21, 89, 0, 3, 52,
           38, 44, 93, 36, 57, 90, 98, 58, 9, 50, 72, 87, 1, 69, 55, 46]
)


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
    def load_ckpt(ckpt_path):
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint path {} does not exists!'.format(ckpt_path))
        state = checkpoints.restore_checkpoint(ckpt_path, None)
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
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        x_proto, y_proto = load_ckpt(ckpt_path)
    return x_proto, y_proto


def main(dataset_name, data_path=None, zca_path=None, ckpt_dir=None, ckpt_name='', random_seed=0,
         eval_batch_size=1000, arch='conv', width=128, depth=3, normalization='identity', pooling='avg',
         use_chunk=False, chunk_size=2000, use_cl=False, cl_steps=0, cl_step_idx=0,
         cl_seed=0, optimizer='adam', learning_rate=0.0003, weight_decay=0.0003, loss='mse', temperature=1.0,
         num_eval=5, num_online_eval_updates=10000):
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
    (x_train, y_train, x_test, y_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset,
                                                                                                    return_raw=True)
    cls_idx_train = {i: [] for i in range(100)}
    cls_idx_test = {i: [] for i in range(100)}

    for idx in range(y_train.shape[0]):
        cls_idx_train[y_train[idx]].append(idx)
    for idx in range(y_test.shape[0]):
        cls_idx_test[y_test[idx]].append(idx)

    # Get CL dataset
    cls_per_step = 100 // cl_steps
    assert cl_step_idx < cl_steps, f'cl_step_idx {cl_step_idx} should be smaller than cl_steps {cl_steps}'

    print(class_order['seed{}'.format(cl_seed)])
    class_subset = class_order['seed{}'.format(cl_seed)][:(cl_step_idx + 1) * cls_per_step]
    logging.info('Class subset: {}'.format(class_subset))
    train_idx = [idx for cls in class_subset for idx in cls_idx_train[cls]]
    test_idx = [idx for cls in class_subset for idx in cls_idx_test[cls]]

    x_train = x_train[train_idx, :]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx, :]
    y_test = y_test[test_idx]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    logging.info('Train size {}'.format(len(ds_train)))

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

    x_proto, y_proto = load_proto(ckpt_dir, ckpt_name, use_chunk=use_chunk, chunk_size=chunk_size, use_cl=use_cl,
                                  steps=cl_steps, idx=cl_step_idx, seed=cl_seed)

    logging.info('x_proto shape {}, y_proto shape {}'.format(x_proto.shape, y_proto.shape))

    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    workdir = ckpt_dir.replace('train_log', 'eval_log')

    hparam = '{}_w{}_{}/{}_lr{}_wd{}_aug{}'.format(arch, width, normalization, optimizer, learning_rate,
                                                   weight_decay, aug_strategy)
    workdir = os.path.join(workdir, hparam)
    writer = metric_writers.create_default_writer(logdir=workdir)
    logging.info('Working directory: {}'.format(workdir))

    res_dir = os.path.join(ckpt_dir, 'summary')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

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
    steps_per_log = num_online_eval_updates / 2

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

    df = pd.DataFrame([[dataset_name, ckpt_dir, ckpt_name, arch, width, depth, cl_steps, cl_step_idx, cl_seed,
                        np.mean(acc), np.std(acc), datetime.now().strftime("%y%m%d %H:%M:%S")]],
                      columns=['dataset', 'ckpt_dir', 'ckpt_name', 'arch', 'width', 'depth', 'cl_steps', 'cl_step_idx',
                               'cl_seed', 'mean', 'std', 'timestamp'])

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
