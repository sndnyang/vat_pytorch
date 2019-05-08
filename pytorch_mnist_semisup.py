import os
import time
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from torch_func.utils import *
from torch_func.load_dataset import load_dataset
from torch_func.evaluate import evaluate_classifier
from torch_func.model import MLP
from torch_func.vat import VAT


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Playground', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--trainer', type=str, default='vat', metavar='S', help='vat default use vat, mle will not use unlabeled data')
    parser.add_argument('--size', type=int, default=100, help='size of training data set 0=all(default: 100)')
    parser.add_argument('--num-valid', type=int, default=1000, help='size of validation set (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 1000)')
    parser.add_argument('--num-batch-it', type=int, default=500, metavar='N', help='number of batch iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="5", metavar='N', help='gpu id list (default: 5)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training data set (default: 100)')
    parser.add_argument('--ul-batch-size', type=int, default=250, help='size of training data set 0=all(default: 100)')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for VAT (default: 0.3)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT (default: 1e-6)')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm use affine')
    parser.add_argument('--top-bn', action='store_true', default=False, help='use top batch norm')
    parser.add_argument('--entmin', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory')
    parser.add_argument('--log-arg', type=str, default='trainer-eps-xi-affine-top_bn', metavar='S', help='show the arguments in directory name')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')

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

    if args.debug:
        args.num_batch_it = 250
        args.log_arg += "-debug"
        args.log_interval = 1
        args.num_epochs = 3

    args_dict = vars(args)
    run_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    exp_marker = "-".join("%s=%s" % (e, str(args_dict.get(e, "None"))) for e in args.log_arg.split("-"))
    exp_marker = "VAT-semi/mnist/%s_%s" % (exp_marker, run_time)
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    args.dir_path = dir_path
    print("args in this experiment %s", '\n'.join(str(e) for e in sorted(vars(args).items())))
    args.writer = SummaryWriter(log_dir=dir_path)
    return args, kwargs


def set_framework_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def train():
    """Training function."""
    # select uniform labeled data, need numpy seed
    # Do shuffling on uniform labeled data in load_dataset with a certain seed of numpy. It works for both of PyTorch and Theano
    np.random.seed(args.seed)
    train_l, train_ul, test_set = load_dataset("mnist", size=args.size)
    print("N_train_labeled:{}, N_train_unlabeled:{}".format(train_l.N, train_ul.N))

    test_set = TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = DataLoader(test_set, 128, False)

    batch_size_l = args.batch_size
    batch_size_ul = args.ul_batch_size

    n_train = train_l.label.shape[0]
    n_ul_train = train_ul.label.shape[0]

    # set_framework_seed(args.seed)
    np.random.seed(args.seed)
    model = MLP(affine=args.affine, top_bn=args.top_bn)
    model.apply(weights_init)            # weights init is based on numpy, so only need np.random.seed()
    model = model.to(args.device)

    # Define losses.
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    model.train()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    l_i, ul_i = 0, 0

    for epoch in range(args.num_epochs):
        # for debugging/comparison with Theano, just disable the random shuffling
        # remove the randomness on data shuffling while the accuracy keeps the same

        # rand_ind = np.random.permutation(train_l.label.shape[0])
        rand_ind = np.arange(train_l.label.shape[0])
        train_images = train_l.data[rand_ind]
        train_labels = train_l.label[rand_ind]
        # rand_ind = np.random.permutation(train_ul.data.shape[0])
        rand_ind = np.arange(train_ul.label.shape[0])
        train_ul_images = train_ul.data[rand_ind]
        for i in range(args.num_batch_it):

            images = torch.FloatTensor(train_images[batch_size_l*l_i:batch_size_l*(l_i + 1)])
            labels = torch.LongTensor(train_labels[batch_size_l*l_i:batch_size_l*(l_i + 1)])
            images, labels = images.to(args.device), labels.to(args.device)

            ul_images = torch.FloatTensor(train_ul_images[batch_size_ul*ul_i:batch_size_ul*(ul_i + 1)])
            ul_images = ul_images.to(args.device)

            # the same as theano code. Only hold when total_size % batch_size == 0, 100 % 100 == 0 and 59000 % 250 == 0
            l_i = 0 if l_i >= n_train / batch_size_l - 1 else l_i + 1
            ul_i = 0 if ul_i >= n_ul_train / batch_size_ul - 1 else ul_i + 1

            logits = model(images)

            total_loss = 0
            sup_loss = 0

            # supervised loss
            xent_loss = criterion(logits, labels)
            sup_loss += xent_loss

            unsup_loss = 0
            vat_criterion = VAT(args.device, eps=args.eps, xi=args.xi, use_entmin=args.entmin, debug=args.debug)
            unsup_loss += vat_criterion(model, ul_images)
            if args.trainer == "mle":
                total_loss += sup_loss
            else:
                total_loss += sup_loss + unsup_loss

            if (epoch == 0 and i < 50) or (i % 50 == 0):
                print("iteration %d" % i)
                print("total loss %g" % total_loss)
                print("sup %g" % sup_loss)
                print("unsup %g" % unsup_loss)
                print("labeled data %g" % images.sum())
                print("unlabeled data %g" % ul_images.sum())
                n_err, val_loss = evaluate_classifier(model, test_loader, args.device)
                acc = 1 - n_err / len(test_set)
                args.writer.add_scalar("Train/iter_xent_loss", xent_loss, epoch * args.num_batch_it + i)
                args.writer.add_scalar("Train/iter_unsup_loss", unsup_loss, epoch * args.num_batch_it + i)
                args.writer.add_scalar("Train/TotalLoss", total_loss, epoch * args.num_batch_it + i)
                args.writer.add_scalar("Test/iter_acc", acc, epoch * args.num_batch_it + i)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        if (epoch % args.log_interval) == 0:
            if scheduler:
                lr = scheduler.get_lr()[0]
                print("learning rate %f" % lr)

            print("Epoch %d SupLoss %.5f, UnsupLoss %.5f, Total Train Loss %.5f" % (epoch, sup_loss, unsup_loss, total_loss))
            n_err, val_loss = evaluate_classifier(model, test_loader, args.device)
            acc = 1 - n_err / len(test_set)
            print('Epoch {} Acc {:.4} val/test loss {:.5}'.format(epoch, acc, val_loss))
            pred_y = torch.max(logits, dim=1)[1]
            train_acc = 1.0 * (pred_y == labels).sum().item() / pred_y.shape[0]
            args.writer.add_scalar("Train/xent_loss", xent_loss, epoch)
            args.writer.add_scalar("Train/unsup_loss", unsup_loss, epoch)
            args.writer.add_scalar("Train/total_loss", total_loss, epoch)
            args.writer.add_scalar("Test/Acc", acc, epoch * args.num_batch_it)
            args.writer.add_scalar("Test/EpochAcc", acc, epoch)
            args.writer.add_scalar("Train/Acc", train_acc, epoch)
            if scheduler:
                args.writer.add_scalar("optimizer/learning_rate", lr, epoch)


if __name__ == "__main__":
    args, kwarg = parse_args()
    train()

