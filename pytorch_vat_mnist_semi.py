import os
import time
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from ExpUtils import wlog
from torch_func.utils import *
from torch_func.mnist_load_dataset import load_dataset
from torch_func.evaluate import evaluate_classifier
from torch_func.MLP import MLPSemi
from torch_func.vat import VAT


def parse_args():
    parser = argparse.ArgumentParser(description='VAT Semi-supervised on MNIST', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--trainer', type=str, default='vat', help='vat, mle, (default: vat)')
    parser.add_argument('--size', type=int, default=100, help='size of training data set (default: 100)')
    parser.add_argument('--num-valid', type=int, default=1000, help='size of validation set (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100)')
    parser.add_argument('--num-batch-it', type=int, default=500, metavar='N', help='number of batch iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="5", metavar='N', help='gpu id list (default: 5)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status, (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training data set (default: 100)')
    parser.add_argument('--ul-batch-size', type=int, default=250, help='size of training data set 0=all(default: 250)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay (default: 0.95)')
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for VAT (default: 0.3)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT (default: 1e-6)')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm affine configuration')
    parser.add_argument('--top-bn', action='store_true', default=False, help='enable top batch norm layer')
    parser.add_argument('--ent_min', action='store_true', default=False, help='enable entropy min')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
    parser.add_argument('--log-arg', type=str, default='trainer-eps-xi-lr-top_bn', metavar='S', help='show the arguments in directory name')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')

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
        exp_marker = "VAT-semi/mnist/%s_%s" % (exp_marker, run_time)
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
    # select uniform labeled data, need numpy seed
    # Do shuffling on uniform labeled data in load_dataset with a certain seed of numpy. It works for both of PyTorch and Theano
    # unlabeled set won't contain labeled data
    train_l, train_ul, test_set = load_dataset("mnist", size=args.size)
    # 100, 58900
    wlog("N_train_labeled:{}, N_train_unlabeled:{}".format(train_l.size, train_ul.size))

    test_set = TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = DataLoader(test_set, 128, False)

    batch_size_l = args.batch_size
    batch_size_ul = args.ul_batch_size

    n_train = train_l.label.shape[0]
    n_ul_train = train_ul.label.shape[0]

    model = MLPSemi(affine=args.affine, top_bn=args.top_bn)
    if args.debug:
        # weights init is based on numpy, so only need np.random.seed()
        np.random.seed(args.seed)
        model.apply(weights_init_normal)
    model = model.to(args.device)

    # Define losses.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model.train()

    l_i, ul_i = 0, 0
    vat_criterion = VAT(args.device, eps=args.eps, xi=args.xi, use_ent_min=args.ent_min, debug=args.debug)

    # Training part with the same data loading way of official VAT theano code.
    for epoch in range(args.num_epochs):
        # When comparing with Theano, just disable the random shuffling
        # remove the randomness on data shuffling to make sure theano/PyTorch read the same batch of data
        rand_ind = np.random.permutation(train_l.label.shape[0])
        if args.debug:
            rand_ind = np.arange(train_l.label.shape[0])
        train_images = train_l.data[rand_ind]
        train_labels = train_l.label[rand_ind]
        rand_ind = np.random.permutation(train_ul.data.shape[0])
        if args.debug:
            rand_ind = np.arange(train_ul.label.shape[0])
        train_ul_images = train_ul.data[rand_ind]
        for it in range(args.num_batch_it):

            images = torch.FloatTensor(train_images[batch_size_l*l_i:batch_size_l*(l_i + 1)])
            labels = torch.LongTensor(train_labels[batch_size_l*l_i:batch_size_l*(l_i + 1)])
            images, labels = images.to(args.device), labels.to(args.device)

            # if the length of train_ul_images is 58900 removing 100 labeled data and 1000 validation data,
            # it also works. The size of last batch is 150, instead of ul_batch_size 250
            ul_images = torch.FloatTensor(train_ul_images[batch_size_ul*ul_i:batch_size_ul*(ul_i + 1)])
            ul_images = ul_images.to(args.device)

            # the same as theano code.
            l_i = 0 if l_i >= n_train / batch_size_l - 1 else l_i + 1
            ul_i = 0 if ul_i >= n_ul_train / batch_size_ul - 1 else ul_i + 1

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

            if it % 100 == 0 or (args.debug and it < 50):
                n_err, test_loss = evaluate_classifier(model, test_loader, args.device)
                acc = 1 - n_err / len(test_set)
                wlog("iteration %d, total loss: %g, sup: %g, unlabeled: %g, test ce-loss: %g, test acc: %g" % (it, total_loss, sup_loss, ul_loss, test_loss, acc))
                if args.vis:
                    args.writer.add_scalar("Train/iter_xent_loss", ce_loss, epoch * args.num_batch_it + it)
                    args.writer.add_scalar("Train/iter_unsup_loss", ul_loss, epoch * args.num_batch_it + it)
                    args.writer.add_scalar("Train/TotalLoss", total_loss, epoch * args.num_batch_it + it)
                    args.writer.add_scalar("Test/iter_acc", acc, epoch * args.num_batch_it + it)

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
