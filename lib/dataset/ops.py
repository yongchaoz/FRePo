import tqdm
import functools
from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def blockshaped(arr, h, w, c, nrows, ncols, is_tf=False):
    """
    Return an array of shape (n, nrows * ncols * c) where
    n * nrows * ncols * c= arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    if is_tf:
        arr = tf.reshape(arr, shape=(h // nrows, nrows, w // ncols, ncols, c))
        arr = tf.transpose(arr, perm=[0, 2, 1, 3, 4])
        arr = tf.reshape(arr, shape=(-1, nrows * ncols * c))
        return arr
    else:
        return arr.reshape(h // nrows, nrows, w // ncols, ncols, c).swapaxes(1, 2).reshape(-1, nrows * ncols * c)


def unblockshaped(arr, h, w, c, nrows, ncols, is_tf=False):
    """
    Return an array of shape (h, w, c) where
    h * w * c = arr.size

    If arr is of shape (n, nrows * ncols * c), n sublocks of shape (nrows * ncols * c),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    if is_tf:
        arr = tf.reshape(arr, shape=(h // nrows, w // ncols, nrows, ncols, c))
        arr = tf.transpose(arr, perm=[0, 2, 1, 3, 4])
        arr = tf.reshape(arr, shape=(h, w, c))
        return arr
    else:
        return arr.reshape(h // nrows, w // ncols, nrows, ncols, c).swapaxes(1, 2).reshape(h, w, c)


def checkboardshaped(arr, h, w, c, nrows, ncols, is_tf=False):
    stride_row = h // nrows
    stride_col = w // ncols

    arr = arr.reshape(h, w, c)
    new_arr = jnp.zeros(shape=(stride_row, stride_col, nrows, ncols, c))

    for i in range(h // nrows):
        for j in range(w // ncols):
            new_arr = new_arr.at[i, j, :, :, :].set(arr[i::stride_row, j::stride_col, :])

    return new_arr.reshape(-1, nrows * ncols * c)


def uncheckboardshaped(arr, h, w, c, nrows, ncols, is_tf=False):
    arr = arr.reshape(h // nrows, w // ncols, nrows, ncols, c)
    new_arr = jnp.zeros(shape=(h, w, c))

    for i in range(h // nrows):
        for j in range(w // ncols):
            new_arr = new_arr.at[i::h // nrows, j::w // ncols, :].set(arr[i, j, :, :, :])

    return new_arr


def center_crop(x, resolution):
    shape = tf.shape(x)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    x = tf.slice(x, begin, [size, size, 3])
    x = tf.image.resize_with_pad(x, resolution, resolution, method='area', antialias=True)
    return x


def compute_zca_mean_cov_ds(ds, img_shape, mean=None, std=None, resolution=32, block_size=None, batch_size=1000,
                            use_checkboard=False):
    rows = img_shape[0] if img_shape[0] is not None else resolution
    cols = img_shape[1] if img_shape[1] is not None else resolution
    channels = img_shape[2] if img_shape[2] is not None else 3
    dim = rows * cols * channels

    ds = ds.map(lambda x, y: tf.cast(x, dtype='float32') / 255.0, tf.data.AUTOTUNE)

    if None in img_shape:
        ds = ds.map(lambda x: center_crop(x, resolution), tf.data.AUTOTUNE)

    if mean is not None:
        ds = ds.map(lambda x: (x - mean) / std, tf.data.AUTOTUNE)

    ds = ds.map(lambda x: tf.reshape(x, shape=(dim,)), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    zca_sum = jnp.zeros(shape=(dim,))
    count = 0

    if block_size is not None:
        assert rows % block_size == 0, 'rows ({}) is not evenly divisible by block_size ({})'.format(rows, block_size)
        assert cols % block_size == 0, 'cols ({}) is not evenly divisible by block_size ({})'.format(cols, block_size)
        block_dim = block_size * block_size * channels
        cov_sum = jnp.zeros(shape=(dim // block_dim, block_dim, block_dim))
    else:
        cov_sum = jnp.zeros(shape=(dim, dim))

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute ZCA Mean with batch size: {}'.format(batch_size)):
        zca_sum = zca_sum + jnp.sum(x_batch, axis=0)
        count += x_batch.shape[0]

    zca_mean = 1.0 / count * zca_sum

    if use_checkboard:
        reshape_op = jax.vmap(functools.partial(checkboardshaped, nrows=block_size, ncols=block_size, is_tf=False),
                              in_axes=(0, None, None, None))
    else:
        reshape_op = jax.vmap(functools.partial(blockshaped, nrows=block_size, ncols=block_size, is_tf=False),
                              in_axes=(0, None, None, None))

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute ZCA Covariance with batch size: {}'.format(batch_size)):
        x_batch = x_batch - zca_mean
        if block_size is not None:
            x_batch = reshape_op(x_batch, rows, cols, channels)
            cov_sum = cov_sum + jnp.einsum('ijk,ijl->jkl', x_batch, x_batch)
        else:
            cov_sum = cov_sum + x_batch.T.dot(x_batch)

    cov = 1.0 / count * cov_sum

    logging.info('Total number of data: {}, ZCA Mean shape: {}, ZCA Covariance shape: {}'.format(count, zca_mean.shape,
                                                                                                 cov.shape))

    return zca_mean, cov


def compute_channel_mean_std_ds(ds, img_shape, resolution=32, batch_size=1000):
    if None in img_shape:
        dim = resolution * resolution
    else:
        dim = functools.reduce(lambda x, y: x * y, img_shape[:-1], 1)

    ds = ds.map(lambda x, y: tf.cast(x, dtype='float32') / 255.0, tf.data.AUTOTUNE)
    if None in img_shape:
        ds = ds.map(lambda x: center_crop(x, resolution), tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.reshape(x, shape=(dim, img_shape[-1])), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    mean = jnp.zeros(shape=(img_shape[-1],))
    var = jnp.zeros(shape=(img_shape[-1],))
    count = 0

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute mean with batch size: {}'.format(batch_size)):
        mean = mean + jnp.sum(x_batch, axis=(0, 1))
        count += x_batch.shape[0]

    mean = 1.0 / (count * dim) * mean

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute variance with batch size: {}'.format(batch_size)):
        var = var + jnp.sum(jnp.square(x_batch - mean), axis=(0, 1))

    std = jnp.sqrt(1.0 / (count * dim) * var)

    logging.info('Total number of data: {}, mean: {}, std: {}'.format(count, mean, std))

    return mean, std


def get_whitening_transform(cov, n_train, zca_reg=1e-5, use_mean_block=False):
    def _get_whitening_transform(cov, n_train, zca_reg):
        """Returns 2D matrix that performs whitening transform.
        Whitening transform is a (d,d) matrix (d = number of features) which acts on
        the right of a (n, d) batch of flattened data.
        """
        reg_amount = zca_reg * jnp.trace(cov) / cov.shape[0]

        u, s, _ = jnp.linalg.svd(cov + reg_amount * jnp.eye(cov.shape[0]))
        sqrt_zca_eigs = s ** (1 / 2)
        inv_sqrt_zca_eigs = s ** (-1 / 2)

        # rank control
        if n_train < cov.shape[0]:
            sqrt_zca_eigs = sqrt_zca_eigs.at[n_train:].set(
                jnp.ones(sqrt_zca_eigs[n_train:].shape[0]))
            inv_sqrt_zca_eigs = inv_sqrt_zca_eigs.at[n_train:].set(
                jnp.ones(inv_sqrt_zca_eigs[n_train:].shape[0]))
        rev_whitening_transform = jnp.einsum('ij,j,kj->ik', u, sqrt_zca_eigs, u, optimize=True)
        whitening_transform = jnp.einsum('ij,j,kj->ik', u, inv_sqrt_zca_eigs, u, optimize=True)
        return whitening_transform, rev_whitening_transform, reg_amount, sqrt_zca_eigs, inv_sqrt_zca_eigs

    get_transform = functools.partial(_get_whitening_transform, n_train=n_train, zca_reg=zca_reg)
    jit_get_transform = jax.jit(get_transform)

    logging.info('Performing zca whitening preprocessing with reg: %.2e', zca_reg)
    if len(cov.shape) == 3 and use_mean_block:
        logging.info('Use mean block!')
        cov = jnp.mean(cov, axis=0)

    if len(cov.shape) == 3:
        whitening_transform = []
        rev_whitening_transform = []
        # Sequential form, otherwise may get OOM
        for i in range(cov.shape[0]):
            a, b, c, d, e = jit_get_transform(cov[i])
            whitening_transform.append(a)
            rev_whitening_transform.append(b)

            logging.info('Raw zca regularization strength: {}'.format(c))
            logging.info('sqrt_zca_eigs: {}, {}'.format(d.shape, d))
            logging.info('inv_sqrt_zca_eigs: {}, {}'.format(e.shape, e))

        whitening_transform = jnp.stack(whitening_transform)
        rev_whitening_transform = jnp.stack(rev_whitening_transform)
    else:
        whitening_transform, rev_whitening_transform, c, d, e = jit_get_transform(cov)
        logging.info('Raw zca regularization strength: {}'.format(c))
        logging.info('sqrt_zca_eigs: {}, {}'.format(d.shape, d))
        logging.info('inv_sqrt_zca_eigs: {}, {}'.format(e.shape, e))

    return whitening_transform, rev_whitening_transform


def get_preprocess_op_np(mean=None, std=None, zca_mean=None, whitening_transform=None, rev_whitening_transform=None,
                         block_size=None, use_mean_block=False, use_checkboard=False):
    if use_checkboard:
        reshape_op = jax.vmap(
            functools.partial(checkboardshaped, nrows=block_size, ncols=block_size, is_tf=False),
            in_axes=(0, None, None, None))
        unreshape_op = jax.vmap(
            functools.partial(uncheckboardshaped, nrows=block_size, ncols=block_size, is_tf=False),
            in_axes=(0, None, None, None))
    else:
        reshape_op = jax.vmap(functools.partial(blockshaped, nrows=block_size, ncols=block_size, is_tf=False),
                              in_axes=(0, None, None, None))
        unreshape_op = jax.vmap(functools.partial(unblockshaped, nrows=block_size, ncols=block_size, is_tf=False),
                                in_axes=(0, None, None, None))

    # This operation deals with a batch of data per time
    def preprocess_op(images):
        if mean is not None:
            images = (images - mean) / std
        if zca_mean is not None:
            orig_shape = images.shape
            images = images.reshape(orig_shape[0], -1)
            images = images - zca_mean
            if block_size is not None:
                images = reshape_op(images, orig_shape[-3], orig_shape[-2], orig_shape[-1])
                if use_mean_block:
                    images = jnp.einsum('...j,jk->...k', images, whitening_transform)
                else:
                    images = jnp.einsum('...ij,ijk->...ik', images, whitening_transform)
                images = unreshape_op(images, orig_shape[-3], orig_shape[-2], orig_shape[-1])
            else:
                images = jnp.einsum('...j,jk->...k', images, whitening_transform)
            images = images.reshape(orig_shape)
        return images

    def preprocess_op_rev(images):
        if zca_mean is not None:
            orig_shape = images.shape
            images = images.reshape(orig_shape[0], -1)
            if block_size is not None:
                images = reshape_op(images, orig_shape[-3], orig_shape[-2], orig_shape[-1])
                if use_mean_block:
                    images = jnp.einsum('...j,jk->...k', images, rev_whitening_transform)
                else:
                    images = jnp.einsum('...ij,ijk->...ik', images, rev_whitening_transform)
                images = unreshape_op(images, orig_shape[-3], orig_shape[-2], orig_shape[-1])
            else:
                images = jnp.einsum('...j,jk->...k', images, rev_whitening_transform)
            images = images.reshape(orig_shape[0], -1)
            images = images + zca_mean
            images = images.reshape(orig_shape)
        if mean is not None:
            images = images * std + mean
        return images

    return preprocess_op, preprocess_op_rev


def get_preprocess_op_tf(resize=False, resolution=None, mean=None, std=None, zca_mean=None, whitening_transform=None,
                         block_size=None):
    # This operation deals with one data per time
    def preprocess_op(x):
        if resize:
            shape = tf.shape(x)
            h, w = shape[0], shape[1]
            size = tf.minimum(h, w)
            begin = tf.cast([h - size, w - size], tf.float32) / 2.0
            begin = tf.cast(begin, tf.int32)
            begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
            x = tf.slice(x, begin, [size, size, 3])
            x = tf.image.resize_with_pad(x, resolution, resolution, method='area', antialias=True)

        x = tf.cast(x, dtype='float32')
        x = x / 255.0

        if mean is not None:
            x = (x - mean) / std
        if zca_mean is not None:
            orig_shape = x.shape
            x = tf.reshape(x, shape=(-1,))
            x = x - zca_mean
            if block_size is not None:
                x = blockshaped(x, orig_shape[-3], orig_shape[-2], orig_shape[-1], block_size, block_size, is_tf=True)
                x = tf.einsum('...ij,ijk->...ik', x, whitening_transform)
                x = unblockshaped(x, orig_shape[-3], orig_shape[-2], orig_shape[-1], block_size, block_size, is_tf=True)
            else:
                x = tf.einsum('...j,jk->...k', x, whitening_transform)

            x = tf.reshape(x, shape=orig_shape)
        return x

    return preprocess_op


def load_data(ds, img_shape, preprocess_op, resolution=32, batch_size=1000):
    size = len(ds)
    logging.info('Dataset size: {}'.format(size))
    if None in img_shape:
        x = np.zeros(shape=(size, resolution, resolution, 3))
    else:
        x = np.zeros(shape=(size, img_shape[0], img_shape[1], img_shape[2]))

    ds = ds.map(lambda x, y: (tf.cast(x, dtype='float32') / 255.0, y), tf.data.AUTOTUNE)
    if None in img_shape:
        ds = ds.map(lambda x, y: (center_crop(x, resolution), y), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    y_list = []
    count = 0
    for x_batch, y_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Process the data'):
        num = x_batch.shape[0]
        x_processed = np.array(preprocess_op(x_batch))
        x[count:count + num] = x_processed
        y_list.append(y_batch)
        count += num

    return x, np.concatenate(y_list, axis=0)


def write_tfrecord(ds, filepattern, preprocess_op, features, num_per_shard, num_shard):
    count = 0
    shard = 0
    shard_lengths = []

    writer = tf.io.TFRecordWriter(
        '{}.tfrecord-{}-of-{}'.format(filepattern, str(shard).zfill(5), str(num_shard).zfill(5)))

    for x_batch, y_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Process the data'):
        x_processed = np.array(preprocess_op(x_batch), dtype=np.float32)

        for i in range(x_processed.shape[0]):
            data = {'image': x_processed[i], 'label': y_batch[i]}
            writer.write(features.serialize_example(data))
            count += 1

            if count == num_per_shard:
                shard_lengths.append(count)
                writer.flush()
                writer.close()
                count = 0
                shard += 1
                if shard < num_shard:
                    writer = tf.io.TFRecordWriter(
                        '{}.tfrecord-{}-of-{}'.format(filepattern, str(shard).zfill(5), str(num_shard).zfill(5)))

    if count != 0:
        shard_lengths.append(count)
        writer.flush()
        writer.close()

    return shard_lengths


def process2tfrecord(ds_train, ds_test, data_dir, dataset_name, img_shape, num_classes, preprocess_op, resolution=32,
                     batch_size=1000, num_per_shard=10000):
    def get_ds(ds):
        ds = ds.map(lambda x, y: (tf.cast(x, dtype='float32') / 255.0, y), tf.data.AUTOTUNE)
        if None in img_shape:
            ds = ds.map(lambda x, y: (center_crop(
                x, resolution), y), tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    if None in img_shape:
        img_shape = (resolution, resolution, 3)

    features = tfds.features.FeaturesDict({
        'image': tfds.features.Tensor(shape=img_shape, dtype=tf.float32),
        'label': tfds.features.ClassLabel(num_classes=num_classes)})

    # Process train
    size = len(ds_train)
    num_shard = size // num_per_shard
    if size % num_per_shard != 0:
        num_shard += 1

    logging.info('Number of examples: {}, num per shard: {}, num shards: {}'.format(
        size, num_per_shard, num_shard))

    filepattern = '{}/{}-train'.format(data_dir, dataset_name)

    shard_lengths_train = write_tfrecord(
        get_ds(ds_train), filepattern, preprocess_op, features, num_per_shard, num_shard)

    # Process test
    size = len(ds_test)
    num_shard = size // num_per_shard
    if size % num_per_shard != 0:
        num_shard += 1

    logging.info('Number of examples: {}, num per shard: {}, num shards: {}'.format(
        size, num_per_shard, num_shard))

    filepattern = '{}/{}-test'.format(data_dir, dataset_name)

    shard_lengths_test = write_tfrecord(
        get_ds(ds_test), filepattern, preprocess_op, features, num_per_shard, num_shard)

    split_infos = [
        tfds.core.SplitInfo(
            name='train', shard_lengths=shard_lengths_train, num_bytes=0),
        tfds.core.SplitInfo(
            name='test', shard_lengths=shard_lengths_test, num_bytes=0),
    ]

    tfds.folder_dataset.write_metadata(data_dir=data_dir, features=features, split_infos=split_infos,
                                       supervised_keys=('image', 'label'))
