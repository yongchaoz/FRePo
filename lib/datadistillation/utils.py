import os
import logging

import numpy as np

import flax

import matplotlib.pyplot as plt

from typing import (Any, Tuple, Iterable, Union)

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

Axes = Union[int, Iterable[int]]


def save_proto_np(proto_state, step, image_dir=None, use_pmap=False):
    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.params['x_proto'], proto_state.params['y_proto']

    path = os.path.join(image_dir, 'np')
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
    np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)
    logging.info('Save prototype to numpy! Path: {}'.format(save_path))


def load_proto_np(path):
    npzfile = np.load('{}.npz'.format(path))
    return npzfile['image'], npzfile['label']


def save_frepo_image(proto_state, step, num_classes=10, class_names=None, rev_preprocess_op=None, image_dir=None,
                    use_pmap=False, is_grey=False, save_np=False, save_img=False):
    def scale_for_vis(img, rev_preprocess_op=None):
        if rev_preprocess_op:
            img = rev_preprocess_op(img)
        else:
            img = img / img.std() * 0.2 + 0.5
        img = np.clip(img, 0, 1)
        return img

    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.apply_fn(variables={'params': proto_state.params})

    if save_np and image_dir:
        path = os.path.join(image_dir, 'np')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)

    x_proto = scale_for_vis(x_proto, rev_preprocess_op)

    total_images = y_proto.shape[0]
    total_index = list(range(total_images))
    total_img_per_class = total_images // num_classes
    img_per_class = 100 // num_classes

    if num_classes <= 100:
        select_idx = []
        # always select the top to make it consistent
        for i in range(num_classes):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][:img_per_class]
            select_idx.extend(select)
    else:
        select_idx = []
        # always select the top to make it consistent
        for i in range(100):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][0]
            select_idx.append(select)

    row, col = len(select_idx) // 10, 10
    fig = plt.figure(figsize=(33, 33))

    for i, idx in enumerate(select_idx[: row * col]):
        img = x_proto[idx]
        ax = plt.subplot(row, col, i + 1)
        if class_names is not None:
            ax.set_title('{}'.format(class_names[y_proto[idx].argmax(-1)], y_proto[idx].argmax(-1)), x=0.5, y=0.9,
                         backgroundcolor='silver')
        else:
            ax.set_title('class_{}'.format(y_proto[idx].argmax(-1)), x=0.5, y=0.9, backgroundcolor='silver')

        if is_grey:
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(np.squeeze(img))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])

    fig.patch.set_facecolor('black')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    if save_img and image_dir:
        path = os.path.join(image_dir, 'png')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        fig.savefig('{}.png'.format(save_path), bbox_inches='tight')

    return fig
