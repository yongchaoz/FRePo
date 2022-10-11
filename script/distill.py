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
         num_prototypes_per_class=10, random_seed=0, num_train_steps=None, max_online_updates=100, num_nn_state=10):
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

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset)

    num_prototypes = num_prototypes_per_class * config.dataset.num_classes
    config.kernel.num_prototypes = num_prototypes

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

    exp_name = os.path.join('{}'.format(dataset_name),
                            'step{}K_num{}'.format(num_train_steps // 1000, num_prototypes),
                            '{}_w{}_d{}_{}_ll{}'.format(config.online.arch, config.online.width,
                                                        config.online.depth, config.online.normalization,
                                                        learn_label),
                            'state{}_reset{}'.format(num_nn_state, max_online_updates))

    image_dir = os.path.join(config.train_img, exp_name)
    workdir = os.path.join(config.train_log, exp_name)
    writer = metric_writers.create_default_writer(logdir=workdir)
    logging.info('Working directory: {}'.format(workdir))

    if save_image:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        logging.info('image directory: {}'.format(image_dir))
        if dataset_name in ['mnist', 'fashion_mnist']:
            image_saver = partial(save_frepo_image, num_classes=config.dataset.num_classes,
                                  class_names=config.dataset.class_names, rev_preprocess_op=rev_preprocess_op,
                                  image_dir=image_dir, is_grey=True, save_img=True, save_np=False)
        else:
            image_saver = partial(save_frepo_image, num_classes=config.dataset.num_classes,
                                  class_names=config.dataset.class_names, rev_preprocess_op=rev_preprocess_op,
                                  image_dir=image_dir, is_grey=False, save_img=True, save_np=False)
    else:
        image_saver = None

    # --------------------------------------
    # Creat model
    # --------------------------------------
    if dataset_name in ['mnist', 'fashion_mnist']:
        use_flip = False
        aug_strategy = 'color_crop_rotate_translate_cutout'
    else:
        use_flip = True
        aug_strategy = 'flip_color_crop_rotate_translate_cutout'

    if dataset_name == 'tiny_imagenet':
        if num_prototypes_per_class == 1:
            use_flip = True
            num_online_eval_updates = 1000
        elif num_prototypes_per_class == 10:
            use_flip = False
            num_online_eval_updates = 5000
        else:
            raise ValueError(
                'Unsupported prototypes per class {} for {}'.format(num_prototypes_per_class, dataset_name))

    elif dataset_name == 'imagenet_resized/64x64':
        use_flip = False
        if num_prototypes_per_class == 1:
            num_online_eval_updates = 2500
        elif num_prototypes_per_class == 2:
            num_online_eval_updates = 5000
        else:
            raise ValueError(
                'Unsupported prototypes per class {} for {}'.format(num_prototypes_per_class, dataset_name))
    else:
        num_online_eval_updates = 1000

    if normalization == 'batch':
        has_bn = True
        eval_normalization = 'identity'
    else:
        has_bn = False
        eval_normalization = normalization

    x_proto, y_proto = init_proto(ds_train, num_prototypes_per_class, num_classes=config.dataset.num_classes,
                                  seed=random_seed, scale_y=True)

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
                                           writer=writer, image_saver=image_saver)

    save_proto_np(proto_state, step=num_train_steps, image_dir=image_dir, use_pmap=False)

    logging.info('Finish!')


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
