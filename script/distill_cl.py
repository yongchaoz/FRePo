import sys

sys.path.append("..")

import os
import fire
import ml_collections
from functools import partial

import jax
from absl import logging
import tensorflow as tf

from lib.dataset.dataloader import get_dataset, configure_dataloader
from lib.models.utils import create_model
from lib.datadistillation.utils import save_frepo_image, save_proto_np
from lib.datadistillation.frepo import proto_train_and_evaluate, init_proto, ProtoHolder
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


def main(dataset_name, data_path=None, zca_path=None, train_log=None, train_img=None, save_image=True,
         arch='conv', width=128, depth=3, normalization='identity', learn_label=True,
         num_prototypes_per_class=10, random_seed=0, num_train_steps=None, max_online_updates=100, num_nn_state=10,
         num_online_eval_updates=1000, cl_steps=10, cl_step_idx=0, cl_seed=0, num_eval=1):
    # --------------------------------------
    # Setup
    # --------------------------------------
    config = get_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'

    if not os.path.exists(train_log):
        os.makedirs(train_log)
    if not os.path.exists(train_img):
        os.makedirs(train_img)

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
    class_subset = class_order['seed{}'.format(cl_seed)][
                   cl_step_idx * cls_per_step:(cl_step_idx + 1) * cls_per_step]
    logging.info('Class subset: {}'.format(class_subset))
    train_idx = [idx for cls in class_subset for idx in cls_idx_train[cls]]
    test_idx = [idx for cls in class_subset for idx in cls_idx_test[cls]]

    x_train = x_train[train_idx, :]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx, :]
    y_test = y_test[test_idx]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    num_prototypes = int(num_prototypes_per_class * len(class_subset))
    config.kernel.num_prototypes = num_prototypes

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

    # --------------------------------------
    # Logging
    # --------------------------------------
    steps_per_epoch = config.dataset.train_size // config.kernel.batch_size

    exp_name = os.path.join(dataset_name, 'steps{}_idx{}_seed{}'.format(cl_steps, cl_step_idx, cl_seed))

    image_dir = os.path.join(config.train_img, exp_name)
    workdir = os.path.join(config.train_log, exp_name)
    writer = metric_writers.create_default_writer(logdir=workdir)
    logging.info('Working directory: {}'.format(workdir))

    if save_image:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        logging.info('image directory: {}'.format(image_dir))
        image_saver = partial(save_frepo_image, num_classes=config.dataset.num_classes,
                              class_names=config.dataset.class_names, rev_preprocess_op=rev_preprocess_op,
                              image_dir=image_dir, is_grey=False, save_img=True, save_np=False)
    else:
        image_saver = None

    # --------------------------------------
    # Creat model
    # --------------------------------------
    use_flip = True
    aug_strategy = 'flip_color_crop_rotate_translate_cutout'

    if normalization == 'batch':
        has_bn = True
        eval_normalization = 'identity'
    else:
        has_bn = False
        eval_normalization = normalization

    x_proto, y_proto = init_proto(ds_train, num_prototypes_per_class, num_classes=config.dataset.num_classes,
                                  class_subset=class_subset, seed=cl_seed, scale_y=True)
    logging.info('x_proto shape {}, y_proto shape {}'.format(x_proto.shape, y_proto.shape))

    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    # Centered one hot label for mse loss
    y_transform = lambda y: tf.one_hot(y, config.dataset.num_classes, on_value=1 - 1 / config.dataset.num_classes,
                                       off_value=-1 / config.dataset.num_classes)
    ds_train = configure_dataloader(ds_train, batch_size=config.kernel.batch_size, y_transform=y_transform,
                                    train=True, shuffle=True)
    ds_test = configure_dataloader(ds_test, batch_size=config.kernel.eval_batch_size, y_transform=y_transform,
                                   train=False, shuffle=False)
    dataset = (ds_train, ds_test)

    online_model = create_model(arch, config.dataset.num_classes, width=config.online.width, depth=config.online.depth,
                                normalization=normalization, output=config.online.output)

    eval_model = create_model(arch, config.dataset.num_classes, width=config.online.width, depth=config.online.depth,
                              normalization=eval_normalization, output=config.online.output)

    create_online_state = partial(create_train_state, config=config.online, model=online_model,
                                  learning_rate_fn=lambda x: config.online.learning_rate, has_bn=has_bn)
    create_eval_state = partial(create_train_state, config=config.online, model=eval_model,
                                learning_rate_fn=lambda x: config.online.learning_rate, has_bn=False)

    ph = ProtoHolder(x_proto, y_proto, num_prototypes, learn_label=learn_label)

    proto_state = proto_train_and_evaluate(config.kernel, ph, create_online_state, create_eval_state, dataset,
                                           workdir, seed=random_seed, num_nn_state=num_nn_state,
                                           num_online_eval_updates=num_online_eval_updates,
                                           num_train_steps=num_train_steps, diff_aug=diff_aug, use_flip=use_flip,
                                           max_online_updates=max_online_updates,
                                           steps_per_epoch=steps_per_epoch,
                                           steps_per_log=500, steps_per_eval=10000,
                                           steps_per_checkpoint=1000, save_ckpt=num_train_steps,
                                           steps_per_save_image=num_train_steps // 10, has_bn=has_bn, use_pmap=use_pmap,
                                           writer=writer, image_saver=image_saver, num_eval=num_eval)

    save_proto_np(proto_state, step=num_train_steps, image_dir=image_dir, use_pmap=False)

    logging.info('Finish!')


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
