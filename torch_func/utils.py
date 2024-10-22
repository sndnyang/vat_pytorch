import os
import sys
import random

import torch
import numpy as np
import torch.nn.functional as nfunc
from torch.nn.parameter import Parameter


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


def set_framework_seed(seed, debug=False):
    if debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)
        # if use multi-GPUs, maybe it's required
        # torch.cuda.manual_seed_all(seed)


def weights_init_uniform(m):
    """
    initialize normal distribution weight matrix
    and set bias to 0
    :param m:
    :return:
    """
    class_name = m.__class__.__name__
    fan_in = 0
    if class_name.find('Conv') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1] * shape[2] * shape[3]
    if class_name.find('Linear') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1]
    if fan_in:
        s = 1.0 * np.sqrt(6.0 / fan_in)
        transpose = np.random.uniform(-s, s, m.weight.data.shape).astype("float32")
        tensor = torch.from_numpy(transpose)
        m.weight = Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_normal(m):
    """
    initialize normal distribution weight matrix
    and set bias to 0
    :param m:
    :return:
    """
    class_name = m.__class__.__name__
    fan_in = 0
    if class_name.find('Conv') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1] * shape[2] * shape[3]
    if class_name.find('Linear') != -1:
        shape = m.weight.data.shape
        fan_in = shape[1]
    if fan_in:
        s = 1.0 * np.sqrt(1.0 / fan_in)
        # in PyTorch default shape is [1200, 784]
        # compare to theano, shape is [784, 1200], I do transpose in theano for getting same outputs
        transpose = np.random.normal(0, s, m.weight.data.shape[::-1]).astype("float32").T
        tensor = torch.from_numpy(transpose)
        # print(shape, transpose.sum())
        m.weight = Parameter(tensor, requires_grad=True)
        if m.bias is not None:
            m.bias.data.zero_()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate from start_epoch linearly to zero at the end"""
    if epoch < args.epoch_decay_start:
        return args.lr
    lr = float(args.num_epochs - epoch) / (args.num_epochs - args.epoch_decay_start) * args.lr
    if args.dataset == "cifar10":
        lr *= 0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (0.5, 0.999)
    return lr


def load_checkpoint_by_marker(args, exp_marker):
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    c = 0
    file_name = ""
    # example 060708/80 -> dir path contains 060708 and model_80.pth
    parts = args.resume.split("@")
    for p in os.listdir(dir_path):
        if parts[0] in p:
            c += 1
            if c == 2:
                print("can't resume, find 2")
                sys.exit(-1)
            file_name = os.path.join(dir_path, p, "model.pth" if len(parts) == 1 else "model_%s.pth" % parts[1])
    if file_name == "":
        print("can't resume, find 0")
        sys.exit(-1)
    checkpoint = torch.load(file_name)
    return checkpoint
