import sys

sys.path.append("..")

import os
import fire
import json
import tqdm
import ml_collections
from functools import partial

import jax
from absl import logging
import numpy as np
import tensorflow as tf

from lib.dataset.dataloader import get_dataset, configure_dataloader
from lib.models.utils import create_model
from lib.datadistillation.utils import save_frepo_image, save_proto_np
from lib.datadistillation.frepo import proto_train_and_evaluate, init_proto, ProtoHolder, proto_evaluate
from lib.training.utils import create_train_state, process_batch, pred_step
from lib.dataset.augmax import get_aug_by_name
from lib.training.metrics import mean_squared_loss

from clu import metric_writers

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType


def mia_attack(logits_train, logits_test, loss_train, loss_test, labels_train, labels_test):
    attack_input = AttackInputData(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=labels_train,
        labels_test=labels_test)

    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=False,
        by_percentiles=False,
        by_classification_correctness=False)

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.MULTI_LAYERED_PERCEPTRON,
        AttackType.RANDOM_FOREST,
        AttackType.K_NEAREST_NEIGHBORS
    ]

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types)
    return attacks_result


def predict_on_dataset(ds, nn_state, jit_nn_pred_step):
    logits = []
    loss = []
    labels = []

    for batch in tqdm.tqdm(ds.as_numpy_iterator(), desc='Predict_on_dataset'):
        img, lb = process_batch(batch, use_pmap=False)
        logit = jit_nn_pred_step(nn_state, img)
        logits.append(logit)
        labels.append(lb.argmax(-1))
        loss.append(mean_squared_loss(logit, lb))

    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    loss = np.concatenate(loss, axis=0)
    return logits, labels, loss


def get_attack_res(ds_train, ds_test, nn_state, jit_nn_pred_step):
    logits_train, labels_train, loss_train = predict_on_dataset(
        ds_train, nn_state, jit_nn_pred_step)
    logits_test, labels_test, loss_test = predict_on_dataset(
        ds_test, nn_state, jit_nn_pred_step)
    attacks_result = mia_attack(
        logits_train, logits_test, loss_train, loss_test, labels_train, labels_test)
    return attacks_result


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


def main(dataset_name, data_path=None, zca_path=None, train_log=None, train_img=None, res_dir=None, save_image=True,
         arch='conv', width=128, depth=3, normalization='identity', learn_label=True,
         num_prototypes_per_class=10, random_seed=0, num_train_steps=None, max_online_updates=100, num_nn_state=10,
         num_online_eval_updates=1000, chunk_size=2000, chunk_idx=0):
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
    chunk_num = config.dataset.train_size // chunk_size
    logging.info('Chunk size {}, Chunk_num {}, Chunk_idx {}'.format(chunk_size, chunk_num, chunk_idx))

    assert config.dataset.train_size % chunk_size == 0, 'train_size is not divisible by chunk_size!'

    # Get random chunk dataset
    ds_train, ds_test = get_chunk_ds(x_train, y_train, x_test, y_test, chunk_size, chunk_idx)
    num_prototypes = int(num_prototypes_per_class * config.dataset.num_classes)
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

    exp_name = os.path.join(dataset_name, 'csize{}_cidx{}'.format(chunk_size, chunk_idx))

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

    if normalization == 'batch':
        has_bn = True
        eval_normalization = 'identity'
    else:
        has_bn = False
        eval_normalization = normalization

    x_proto, y_proto = init_proto(ds_train, num_prototypes_per_class, num_classes=config.dataset.num_classes,
                                  seed=random_seed, scale_y=True)
    logging.info('x_proto shape {}, y_proto shape {}'.format(x_proto.shape, y_proto.shape))

    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    # Centered one hot label for mse loss
    y_transform = lambda y: tf.one_hot(y, config.dataset.num_classes, on_value=1 - 1 / config.dataset.num_classes,
                                       off_value=-1 / config.dataset.num_classes)
    ds_mem = configure_dataloader(ds_train, batch_size=config.kernel.eval_batch_size, y_transform=y_transform,
                                  train=False, shuffle=False)
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
                                           writer=writer, image_saver=image_saver, num_eval=2)

    save_proto_np(proto_state, step=num_train_steps, image_dir=image_dir, use_pmap=False)

    # --------------------------------------
    # MIA
    # --------------------------------------
    has_bn = False
    ds_proto = tf.data.Dataset.from_tensor_slices((x_proto, y_proto))
    ds_proto = ds_proto.cache().repeat().shuffle(buffer_size=5000).batch(batch_size=min(y_proto.shape[0], 256))
    jit_nn_pred_step = jax.jit(partial(pred_step, has_feat=True, has_bn=has_bn))

    print('\n==================== MIA ====================\n')
    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    attack_types = ['threshold_attack', 'logistic_regression', 'multi_layered_perceptron', 'random_forest',
                    'k_nearest_neighbors']
    res_proto = dict(
        acc_test=[],
        threshold_attack=[],
        logistic_regression=[],
        multi_layered_perceptron=[],
        random_forest=[],
        k_nearest_neighbors=[],
    )

    res_real = dict(
        acc_test=[],
        threshold_attack=[],
        logistic_regression=[],
        multi_layered_perceptron=[],
        random_forest=[],
        k_nearest_neighbors=[],
    )

    for num_online_eval_updates in [100, 250, 500, 1000, 2000, 4000, 8000, 16000]:
        print('\n==================== Step: {} ====================\n'.format(num_online_eval_updates))
        hparam = '{}_lr{}_steps{}_proto'.format('adam', config.online.learning_rate, num_online_eval_updates)
        workdir_student = os.path.join(workdir, hparam)
        writer = metric_writers.create_default_writer(logdir=workdir_student)
        logging.info('Working directory: {}'.format(workdir_student))

        nn_state_proto, acc_test = proto_evaluate(ds_proto, ds_test, workdir_student, create_eval_state,
                                                  seed=random_seed, num_online_eval_updates=num_online_eval_updates,
                                                  diff_aug=diff_aug, steps_per_log=min(1000, num_online_eval_updates),
                                                  writer=writer, has_bn=has_bn, use_pmap=False)
        res_proto['acc_test'].append(float(np.mean(acc_test)))

        attacks_result_proto = get_attack_res(ds_mem, ds_test, nn_state_proto, jit_nn_pred_step)
        df_proto = attacks_result_proto.calculate_pd_dataframe()

        for a in attack_types:
            res_proto[a].append(float(df_proto[df_proto['attack type'] == a.upper()]['AUC']))
        print(res_proto)
        sys.stdout.flush()

        hparam = '{}_lr{}_steps{}_random'.format('adam', config.online.learning_rate, num_online_eval_updates)

        workdir_student = os.path.join(workdir, hparam)
        writer = metric_writers.create_default_writer(logdir=workdir_student)
        logging.info('Working directory: {}'.format(workdir_student))

        nn_state_random, acc_test = proto_evaluate(ds_train, ds_test, workdir_student, create_eval_state,
                                                   seed=random_seed,
                                                   num_online_eval_updates=num_online_eval_updates, diff_aug=diff_aug,
                                                   steps_per_log=min(1000, num_online_eval_updates),
                                                   writer=writer, has_bn=has_bn, use_pmap=False)
        res_real['acc_test'].append(float(np.mean(acc_test)))
        attacks_result_random = get_attack_res(ds_mem, ds_test, nn_state_random, jit_nn_pred_step)
        df_random = attacks_result_random.calculate_pd_dataframe()
        for a in attack_types:
            res_real[a].append(float(df_random[df_random['attack type'] == a.upper()]['AUC']))
        print(res_real)
        sys.stdout.flush()

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(res_dir, 'csize{}_cidx{}_ll{}.json'.format(chunk_size, chunk_idx, learn_label)),
              'w') as jsonfile:
        json.dump(dict(res_proto=res_proto, res_real=res_real), jsonfile, indent=4)

    logging.info('Finish!')


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
