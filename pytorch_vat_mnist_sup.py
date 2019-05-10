# coding=utf-8

import traceback

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from ExpUtils import *

from torch_func.utils import *
from torch_func.evaluate import evaluate_classifier
from torch_func.mnist_load_dataset import load_mnist_full, load_mnist_for_validation
from torch_func.MLP import MLP
from torch_func.vat import VAT


def parse_args():
    parser = argparse.ArgumentParser(description='VAT supervised learning in PyTorch')
    parser.add_argument('--num-valid', type=int, default=1000, help='size of validation set (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 1000)')
    parser.add_argument('--num-batch-it', type=int, default=500, metavar='N', help='number of batch iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="", metavar='N', help='gpu id list (default: auto select)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training data set (default: 100)')
    parser.add_argument('--layer-sizes', type=str, default='784-1200-600-300-150-10', help='MLP for supervised learning on mnist')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=2.1, help='epsilon for VAT (default: 2.1)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT (default: 1e-6)')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--aug-trans', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--aug-flip', action='store_true', default=False, help='data augmentation flip')
    parser.add_argument('--entmin', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--validation', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')

    args = parser.parse_args()
    args.dir_path = None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.debug:
        args.num_batch_it = 1
        args.vis = True
        args.log_arg += "-debug"
        args.log_interval = 1
        args.num_epochs = 50

    if args.vis:
        exp_marker = "VAT-sup/mnist/eps=%g_xi=%g_min-ent=%s" % (args.eps, args.xi, args.entmin)
        base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
        dir_path = os.path.join(base_dir, exp_marker)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        info = '\n'.join(str(e) for e in sorted(vars(args).items()))
        args.dir_path = dir_path
        args.writer = SummaryWriter(log_dir=dir_path)
        args.writer.add_text("Logs", info, 1)
    wlog("args in this experiment %s", '\n'.join(str(e) for e in sorted(vars(args).items())))
    return args, kwargs


def train(args):
    set_framework_seed(args.seed)
    """Training function."""

    if args.validation:
        train_set, test_set = load_mnist_for_validation(n_v=args.num_validation_samples)
    else:
        train_set, test_set = load_mnist_full()

    test_set = TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = DataLoader(test_set, 128, False)

    # Define losses.
    criterion = nn.CrossEntropyLoss()

    layer_sizes = [int(layer_size) for layer_size in args.layer_sizes.split('-')]
    model = MLP(layer_sizes=layer_sizes)
    set_framework_seed(args.seed)
    model.apply(weights_init_normal)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    # train
    for epoch in range(args.num_epochs):
        for i in range(args.num_batch_it):

            x, t = train_set.get(args.batch_size, gpu=args.gpu_id, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            images = torch.FloatTensor(x)
            labels = torch.LongTensor(t)
            images, labels = images.to(args.device), labels.to(args.device)

            logits = model(images)

            total_loss = 0
            sup_loss = 0

            # supervised loss
            ce_loss = criterion(logits, labels)
            sup_loss += ce_loss

            vat_criterion = VAT(args.device, eps=args.eps, xi=args.xi, use_ent_min=args.entmin)
            ul_loss = vat_criterion(model, images)
            total_loss += sup_loss + ul_loss

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
                    args.writer.add_scalar("Train/ul_loss", ul_loss, epoch)
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
    # noinspection PyBroadException
    try:
        train(arg)
    except KeyboardInterrupt:
        pass
    except BaseException as err:
        traceback.print_exc()
        if arg.dir_path:
            import shutil
            shutil.rmtree(arg.dir_path)
