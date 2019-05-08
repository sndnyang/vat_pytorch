import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tensorboardX import SummaryWriter

from ExpUtils import wlog
from torch_func.utils import set_framework_seed, weights_init_normal
from torch_func.load_dataset import load_dataset
from torch_func.evaluate import evaluate_classifier
import torch_func.CNN as CNN
from torch_func.vat import VAT


def parse_args():
    parser = argparse.ArgumentParser(description='VAT Semi-supervised learning in PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, svhn (default: cifar10)')
    parser.add_argument('--trainer', type=str, default='vat', help='vat, mle, (default: vat)')
    parser.add_argument('--size', type=int, default=4000, help='size of training data set, only support 4000 (default: 4000)')
    parser.add_argument('--arch', type=str, default='CNN9c', help='CNN9 for semi supervised learning on dataset')
    parser.add_argument('--num-valid', type=int, default=1000, help='size of validation set (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100)')
    parser.add_argument('--num-batch-it', type=int, default=400, metavar='N', help='number of batch iterations (default: 400)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="5", metavar='N', help='gpu id list (default: 5)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status, (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size of training data set (default: 32)')
    parser.add_argument('--ul-batch-size', type=int, default=128, help='size of training data set 0=all(default: 128)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay (default: 0.95)')
    parser.add_argument('--eps', type=float, default=10, help='epsilon for VAT, 10 for cifar10, 2.5 for svhn, (default: 10)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT (default: 1e-6)')
    parser.add_argument('--ent_min', action='store_true', default=False, help='enable entropy min')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm affine configuration')
    parser.add_argument('--top-bn', action='store_true', default=False, help='enable top batch norm layer')
    parser.add_argument('--aug-trans', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--aug-flip', action='store_true', default=False, help='data augmentation flip')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='dropout rate, (default: 0.5)')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
    parser.add_argument('--log-arg', type=str, default='trainer-eps-xi-lr-top_bn', metavar='S', help='show the arguments in directory name')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('-r', '--resume', type=str, default='', metavar='S', help='resume from pth file')

    args = parser.parse_args()
    args.dir_path = None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.vis:
        args_dict = vars(args)
        run_time = time.strftime('%d%H%M%S', time.localtime(time.time()))
        exp_marker = "-".join("%s=%s" % (e, str(args_dict.get(e, "None"))) for e in args.log_arg.split("-"))
        exp_marker = "VAT-semi/%s/%s_%s" % (args.dataset, exp_marker, run_time)
        base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
        dir_path = os.path.join(base_dir, exp_marker)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        args.dir_path = dir_path
        args.writer = SummaryWriter(log_dir=dir_path)
    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))
    return args, kwargs


def train(args):
    """Training function."""
    set_framework_seed(args.seed, args.debug)

    train_l, train_ul, test_set = load_dataset("data/%s" % args.dataset, valid=False, dataset_seed=args.seed)
    wlog("N_train_labeled:{}, N_train_unlabeled:{}".format(train_l.size, train_ul.size))

    test_set = TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = DataLoader(test_set, 128, False)

    arch = getattr(CNN, args.arch)
    model = arch(args)
    if args.debug:
        # weights init is based on numpy, so only need np.random.seed()
        np.random.seed(args.seed)
        model.apply(weights_init_normal)

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    # TODO: in VAT, it's a linear scheduler, and setting the mom2
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    model = model.to(args.device)
    model.train()

    # Define losses.
    criterion = nn.CrossEntropyLoss()
    vat_criterion = VAT(args.device, eps=args.eps, xi=args.xi, use_ent_min=args.ent_min, debug=args.debug)

    # train
    for epoch in range(start_epoch, args.num_epochs):
        for it in range(args.num_batch_it):

            x, t = train_l.get(args.batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            images = torch.FloatTensor(x).to(args.device)
            labels = torch.LongTensor(t).to(args.device)

            x_u, t_for_debug = train_ul.get(args.ul_batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            ul_images = torch.FloatTensor(x_u).to(args.device)

            logits = model(images)

            sup_loss = 0

            # supervised loss
            ce_loss = criterion(logits, labels)
            sup_loss += ce_loss

            ul_loss = vat_criterion(model, ul_images)
            if args.trainer == "mle":
                total_loss = sup_loss
            else:
                total_loss = sup_loss + ul_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if ((epoch % args.log_interval) == 0 and it == args.num_batch_it - 1) or (args.debug and it < 50):
                n_err, test_loss = evaluate_classifier(model, test_loader, args.device)
                acc = 1 - n_err / len(test_set)
                wlog("Epoch: %d Train Loss: %.4f ce: %.5f, vat: %.5f, test loss: %.5f, test acc: %.4f" % (epoch, total_loss, ce_loss, ul_loss, test_loss, acc))
                pred_y = torch.max(logits, dim=1)[1]
                train_acc = 1.0 * torch.sum(pred_y == labels).item() / pred_y.shape[0]
                if args.vis:
                    args.writer.add_scalar("Train/xent_loss", ce_loss, epoch)
                    args.writer.add_scalar("Train/unsup_loss", ul_loss, epoch)
                    args.writer.add_scalar("Train/total_loss", total_loss, epoch)
                    args.writer.add_scalar("Train/Acc", train_acc, epoch)
                    args.writer.add_scalar("Test/EpochAcc", acc, epoch)
                    if scheduler:
                        lr = scheduler.get_lr()[0]
                        wlog("learning rate %f" % lr)
                        args.writer.add_scalar("optimizer/learning_rate", lr, epoch)

        if scheduler:
            scheduler.step()


if __name__ == "__main__":
    arg, kwarg = parse_args()
    train(arg)
