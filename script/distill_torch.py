import os
import sys
import fire
import time
import copy
import ml_collections
from tqdm import tqdm
from absl import logging
from functools import partial

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
sys.path.append("../..")

import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch._vmap_internals import vmap

from lib.dataset.dataloader import get_dataset

from lib_torch.utils import get_network, evaluate_synset, get_time, TensorDataset, ParamDiffAug, \
    save_torch_image

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
    config.kernel.optimizer = 'adam'
    config.kernel.learning_rate = 0.001
    config.kernel.batch_size = 1024
    config.kernel.eval_batch_size = 1000

    return config


@vmap
def lb_margin_th(logits):
    dim = logits.shape[-1]
    val, idx = torch.topk(logits, k=2)
    margin = torch.minimum(val[0] - val[1], torch.tensor(1 / dim, dtype=torch.float, device=logits.device))
    return -margin


class SynData(nn.Module):
    def __init__(self, x_init, y_init, learn_label=False):
        super(SynData, self).__init__()
        self.x_syn = nn.Parameter(x_init, requires_grad=True)
        self.y_syn = nn.Parameter(y_init, requires_grad=learn_label)

    def forward(self):
        return self.x_syn, self.y_syn

    def value(self):
        '''Return the synthetic images and labels. Used in deterministic parameterization of synthetic data'''
        return self.x_syn.detach(), self.y_syn.detach()


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class PoolElement():
    def __init__(self, get_model, get_optimizer, get_scheduler, loss_fn, batch_size, max_online_updates, idx, device,
                 step=0):
        self.get_model = get_model
        self.get_optimizer = get_optimizer
        self.get_scheduler = get_scheduler
        self.loss_fn = loss_fn.to(device)
        self.batch_size = batch_size
        self.max_online_updates = max_online_updates
        self.idx = idx
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.initialize()
        self.step = step

    def __call__(self, x, no_grad=False):
        self.model.eval()
        if no_grad:
            with torch.no_grad():
                return self.model(x)
        else:
            return self.model(x)

    def feature(self, x, no_grad=False, weight_grad=False):
        self.model.eval()
        if no_grad:
            with torch.no_grad():
                return self.model.embed(x)
        else:
            self.model.requires_grad_(weight_grad)
            return self.model.embed(x)

    def nfr(self, x_syn, y_syn, x_tar, reg=1e-6, weight_grad=False, use_flip=False):
        if use_flip:
            x_syn_flip = torch.flip(x_syn, dims=[-1])
            x_syn = torch.cat((x_syn, x_syn_flip), dim=0)
            y_syn = torch.cat((y_syn, y_syn), dim=0)

        feat_tar = self.feature(x_tar, no_grad=True)
        feat_syn = self.feature(x_syn, weight_grad=weight_grad)

        kss = torch.mm(feat_syn, feat_syn.t())
        kts = torch.mm(feat_tar, feat_syn.t())
        kss_reg = (kss + np.abs(reg) * torch.trace(kss) * torch.eye(kss.shape[0], device=kss.device) / kss.shape[0])
        pred = torch.mm(kts, torch.linalg.solve(kss_reg, y_syn))
        return pred

    def nfr_eval(self, feat_syn, y_syn, x_tar, kss_reg):
        feat_tar = self.feature(x_tar, no_grad=True)
        kts = torch.mm(feat_tar, feat_syn.t())
        pred = torch.mm(kts, torch.linalg.solve(kss_reg, y_syn))
        return pred

    def train_steps(self, x_syn, y_syn, steps=1):
        self.model.train()
        self.model.requires_grad_(True)
        for step in range(steps):
            x, y = self.get_batch(x_syn, y_syn)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        self.check_for_reset(steps=steps)

    def evaluate_syn(self, x_syn, y_syn):
        pass

    def get_batch(self, xs, ys):
        if ys.shape[0] < self.batch_size:
            x, y = xs, ys
        else:
            sample_idx = np.random.choice(ys.shape[0], size=(self.batch_size,), replace=False)
            x, y = xs[sample_idx], ys[sample_idx]
        return x, y

    def initialize(self):
        self.model = self.get_model().to(self.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer)
        self.step = 0

    def check_for_reset(self, steps=1):
        self.step += steps
        if self.step >= self.max_online_updates:
            self.initialize()


def main(dataset_name, data_path=None, zca_path=None, train_log=None, train_img=None, save_image=True,
         arch='conv', width=128, depth=3, normalization='identity', learn_label=True,
         num_prototypes_per_class=10, random_seed=0, num_train_steps=None, max_online_updates=100, num_nn_state=10):
    config = get_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'

    if not os.path.exists(train_log):
        os.makedirs(train_log)
    if not os.path.exists(train_img):
        os.makedirs(train_img)

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --------------------------------------
    # Dataset
    # --------------------------------------
    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (x_train, y_train, x_test, y_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset,
                                                                                                    return_raw=True)

    im_size = config.dataset.img_shape[0:2]
    channel = config.dataset.img_shape[-1]
    num_classes = config.dataset.num_classes
    class_names = config.dataset.class_names
    class_map = {x: x for x in range(num_classes)}

    x_train = torch.from_numpy(np.transpose(x_train, axes=[0, 3, 1, 2]))
    x_test = torch.from_numpy(np.transpose(x_test, axes=[0, 3, 1, 2]))
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    dst_train = TensorDataset(x_train, y_train)

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
    steps_per_eval = 10000
    steps_per_save = 1000

    lr_syn = config.kernel.learning_rate
    lr_net = config.online.learning_rate
    exp_name = os.path.join('{}'.format(dataset_name),
                            'step{}K_num{}'.format(num_train_steps // 1000, num_prototypes),
                            '{}_w{}_d{}_{}_ll{}'.format(config.online.arch, config.online.width,
                                                        config.online.depth, config.online.normalization,
                                                        learn_label),
                            'state{}_reset{}'.format(num_nn_state, max_online_updates))

    image_dir = os.path.join(config.train_img, exp_name)
    work_dir = os.path.join(config.train_log, exp_name)
    ckpt_dir = os.path.join(work_dir, 'ckpt')
    writer = metric_writers.create_default_writer(logdir=work_dir)
    logging.info('work_dir: {}'.format(work_dir))

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    logging.info("image_dir: {}!".format(image_dir))

    if save_image:
        image_saver = partial(save_torch_image, num_classes=num_classes, class_names=class_names,
                              rev_preprocess_op=rev_preprocess_op, image_dir=image_dir, is_grey=False, save_img=True,
                              save_np=False)
    else:
        image_saver = None

    eval_it_pool = [1, 300, 1000, 3000, 10000]
    model_eval_pool = [arch]

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if normalization == 'batch':
        eval_normalization = 'identity'
    else:
        eval_normalization = normalization

    if dataset_name in ['mnist', 'fashion_mnist']:
        use_flip = False
        aug_strategy = 'color_crop_rotate_scale_cutout'
    else:
        use_flip = True
        aug_strategy = 'flip_color_crop_rotate_scale_cutout'

    if dataset_name == 'tiny_imagenet':
        if num_prototypes_per_class == 1:
            use_flip = True
        elif num_prototypes_per_class == 10:
            use_flip = False
        else:
            raise ValueError(
                'Unsupported prototypes per class {} for {}'.format(num_prototypes_per_class, dataset_name))

    if dataset_name == 'imagenet_resized/64x64':
        use_flip = False

    step_per_prototpyes = {10: 1000, 100: 2000, 200: 20000, 400: 5000, 500: 5000, 1000: 10000, 2000: 40000, 5000: 40000}
    num_online_eval_updates = step_per_prototpyes[num_prototypes]

    args = ml_collections.ConfigDict()
    args.model = arch
    args.device = config.device
    args.lr_net = lr_net
    args.epoch_eval_train = num_online_eval_updates
    args.batch_train = min(num_prototypes, 500)
    args.dsa = True
    args.dsa_strategy = aug_strategy
    args.dsa_param = ParamDiffAug()  # Todo: Implementation is slightly different from JAX Version.

    criterion = nn.MSELoss(reduction='none').to(config.device)
    # --------------------------------------
    # Organize the real dataset
    # --------------------------------------
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images' % (c, len(indices_class[c])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    # --------------------------------------
    # Initialize the synthetic data
    # --------------------------------------
    y_syn = torch.tensor(np.array([np.ones(num_prototypes_per_class) * i for i in range(num_classes)]),
                         dtype=torch.long,
                         device=config.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    x_syn = torch.randn(size=(num_classes * num_prototypes_per_class, channel, im_size[0], im_size[1]),
                        dtype=torch.float)

    for c in range(num_classes):
        x_syn.data[c * num_prototypes_per_class:(c + 1) * num_prototypes_per_class] = get_images(c,
                                                                                                 num_prototypes_per_class).detach().data

    y_scale = np.sqrt(num_classes / 10)
    y_train = F.one_hot(y_train, num_classes=num_classes) - 1 / num_classes
    y_test = F.one_hot(y_test, num_classes=num_classes) - 1 / num_classes

    dst_train = TensorDataset(x_train, y_train)
    dst_test = TensorDataset(x_test, y_test)
    trainloader = InfiniteDataLoader(dst_train, batch_size=1024, shuffle=True, num_workers=0)
    testloader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)

    y_syn = (F.one_hot(y_syn, num_classes=num_classes) - 1 / num_classes) / y_scale

    syndata = SynData(x_syn, y_syn, learn_label=learn_label).to(config.device)
    # Todo: Different from the paper and JAX Version which use LAMB optimizer, Adam is used here.
    synopt = torch.optim.Adam(syndata.parameters(), lr=lr_syn)
    synsch = torch.optim.lr_scheduler.CosineAnnealingLR(synopt, T_max=num_train_steps, eta_min=lr_syn * 0.1)

    step_offset = 0
    best_val_acc = 0.0

    if config.resume:
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
        try:
            checkpoint = torch.load(ckpt_path)
            syndata.load_state_dict(checkpoint['syndata_state_dict'])
            synopt.load_state_dict(checkpoint['synopt_state_dict'])
            synsch.load_state_dict(checkpoint['synsch_state_dict'])
            step_offset = checkpoint['step_offset']
            best_val_acc = checkpoint['best_val_acc']
            logging.info('Load checkpoint from {}!'.format(ckpt_path))
            logging.info('step_offset: {}, best_val_acc: {}'.format(step_offset, best_val_acc))
        except:
            logging.info('No checkpoints found in {}!'.format(ckpt_dir))

    loss_sum = 0.0
    ln_loss_sum = 0.0
    lb_loss_sum = 0.0
    count = 0
    last_t = time.time()

    get_model = lambda: get_network(arch, channel, num_classes, im_size, width=width, depth=depth, norm=normalization)
    get_optimizer = lambda m: torch.optim.Adam(m.parameters(), lr=args.lr_net, betas=(0.9, 0.999))
    get_scheduler = lambda o: torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(o, start_factor=0.01, total_iters=500),
        torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max_online_updates, eta_min=args.lr_net * 0.01)])

    pools = []
    for idx in range(num_nn_state):
        init_step = (max_online_updates // num_nn_state) * idx
        pools.append(PoolElement(get_model=get_model, get_optimizer=get_optimizer, get_scheduler=get_scheduler,
                                 loss_fn=nn.MSELoss(), batch_size=500, max_online_updates=max_online_updates, idx=idx,
                                 device=config.device, step=init_step))

    # --------------------------------------
    # Train
    # --------------------------------------
    for it in range(step_offset + 1, num_train_steps + 1):
        ''' Train synthetic data '''
        x_target, y_target = next(trainloader)
        x_target = x_target.to(config.device)
        y_target = y_target.to(config.device)
        x_syn, y_syn = syndata()

        idx = np.random.randint(low=0, high=num_nn_state)
        pool_m = pools[idx]

        y_pred = pool_m.nfr(x_syn, y_syn, x_target, use_flip=use_flip)
        ln_loss = criterion(y_pred, y_target).sum(dim=-1).mean(0)
        lb_loss = lb_margin_th(y_syn).mean()
        loss = ln_loss + lb_loss

        synopt.zero_grad(set_to_none=True)
        loss.backward()
        synopt.step()
        x_syn, y_syn = syndata.value()
        pool_m.train_steps(x_syn, y_syn, steps=1)

        synsch.step()
        loss_sum += loss.item() * x_target.shape[0]
        ln_loss_sum += ln_loss.item() * x_target.shape[0]
        lb_loss_sum += lb_loss.item() * x_target.shape[0]
        count += x_target.shape[0]

        if it % 100 == 0:
            x_syn, y_syn = syndata.value()
            x_norm = torch.mean(torch.linalg.norm(x_syn.view(x_syn.shape[0], -1), ord=2, dim=-1)).cpu().numpy()
            y_norm = torch.mean(torch.linalg.norm(y_syn.view(y_syn.shape[0], -1), ord=2, dim=-1)).cpu().numpy()
            summary = {'train/loss': loss_sum / count,
                       'train/ln_loss': loss_sum / count,
                       'train/lb_loss': lb_loss_sum / count,
                       'monitor/steps_per_second': count / 1024 / (time.time() - last_t),
                       'monitor/learning_rate': synsch.get_last_lr()[0],
                       'monitor/x_norm': x_norm,
                       'monitor/y_norm': y_norm}
            writer.write_scalars(it, summary)

            last_t = time.time()
            loss_sum, ln_loss_sum, lb_loss_sum, count = 0.0, 0.0, 0.0, 0

        ''' Evaluate synthetic data '''
        if it in eval_it_pool or it % steps_per_eval == 0:
            for model_eval in model_eval_pool:
                print(
                    '----------\nEvaluation\nmodel_train = {}, model_eval = {}, iteration = {}'.format(arch, model_eval,
                                                                                                       it))
                accs = []
                for it_eval in range(3):
                    net_eval = get_network(model_eval, channel, num_classes, im_size, width=width, depth=depth,
                                           norm=eval_normalization).to(
                        config.device)  # get a random model
                    x_syn, y_syn = syndata.value()
                    x_syn_eval, y_syn_eval = copy.deepcopy(x_syn), copy.deepcopy(y_syn)
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, x_syn_eval, y_syn_eval,
                                                             testloader, args)
                    accs.append(acc_test)
                summary = {'eval/acc_mean': np.mean(accs), 'eval/acc_std': np.std(accs)}
                writer.write_scalars(it, summary)

                if float(np.mean(accs)) > best_val_acc:
                    ckpt_path = os.path.join(ckpt_dir, 'best_ckpt.pt')
                    best_val_acc = float(np.mean(accs))
                    torch.save(dict(step_offset=it, best_val_acc=best_val_acc, syndata_state_dict=syndata.state_dict(),
                                    synopt_state_dict=synopt.state_dict(), synsch_state_dict=synsch.state_dict()),
                               ckpt_path)
                    logging.info('{} Save checkpoint to {}, best acc {}!'.format(get_time(), ckpt_path, best_val_acc))

            ''' visualize and save '''
            x_syn, y_syn = syndata.value()
            x_proto, y_proto = copy.deepcopy(x_syn.cpu().numpy()), copy.deepcopy(y_syn.cpu().numpy())
            if image_saver:
                image_saver(x_proto, y_proto, step=it)
            last_t = time.time()

        if it % steps_per_save == 0:
            ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
            torch.save(dict(step_offset=it, best_val_acc=best_val_acc, syndata_state_dict=syndata.state_dict(),
                            synopt_state_dict=synopt.state_dict(), synsch_state_dict=synsch.state_dict()),
                       ckpt_path)
            logging.info('Save checkpoint to {}!'.format(ckpt_path))


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
