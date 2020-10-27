import os
import argparse
import traceback

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_util
import tensorboardX as tbX

from ExpUtils import *
from torch_func.utils import set_framework_seed, weights_init_normal, adjust_learning_rate, load_checkpoint_by_marker
from torch_func.evaluate import evaluate_classifier
from torch_func.load_dataset import load_dataset
from torch_func.mnist_load_dataset import load_dataset as mnist_load_dataset
import models
from Loss import AT, VAT, show_and_save_vat_generated_demo, show_and_save_at_generated_demo


def parse_args():
    parser = argparse.ArgumentParser(description='VAT Supervised learning in PyTorch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, svhn (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='data', help='default: data')
    parser.add_argument('--trainer', type=str, default='vat', help='ce, vat (default: vat)')
    parser.add_argument('--size', type=int, default=0, help='size of training data set, 0 denotes all (default: 0)')
    parser.add_argument('--arch', type=str, default='CNN9', help='CNN9 for semi supervised learning on dataset')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100)')
    parser.add_argument('--num-batch-it', type=int, default=400, metavar='N', help='number of batch iterations (default: 400)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default="", metavar='N', help='gpu id list (default: auto select)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status, (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training data set, MNIST uses 100 (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay used on MNIST (default: 0.95)')
    parser.add_argument('--epoch-decay-start', type=float, default=80, help='start learning rate decay used on SVHN and cifar10 (default: 80)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT loss (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon for VAT loss (default: 1.0)')
    parser.add_argument('--ent-min', action='store_true', default=False, help='use entropy minimum')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm affine configuration')
    parser.add_argument('--top-bn', action='store_true', default=False, help='enable top batch norm layer')
    parser.add_argument('--k', type=int, default=1, help='optimization times, (default: 1)')
    parser.add_argument('--kl', type=int, default=1, help='unlabel loss computing, (default: 1)')
    parser.add_argument('--aug-trans', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--aug-flip', action='store_true', default=False, help='data augmentation flip')
    parser.add_argument('--drop', type=float, default=0.5, help='dropout rate, (default: 0.5)')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
    parser.add_argument('--log-arg', type=str, default='', metavar='S', help='show the arguments in directory name')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('-r', '--resume', type=str, default='', metavar='S', help='resume from pth file')

    args = parser.parse_args()
    args.dir_path = None

    if args.gpu_id == "":
        args.gpu_id = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if not args.log_arg:
        args.log_arg = "trainer-eps-xi-ent_min-top_bn-lr"

    if args.vis:
        args.dir_path = form_dir_path("VAT-sup", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)
        os.mkdir("%s/demo" % args.dir_path)
    wlog("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def get_data(args):
    if args.dataset == "mnist":
        # VAT parameters for MNIST. They are different from SVHN/CIFAR10
        args.batch_size = 100
        args.num_batch_it = 500
        _, train_set, test_set = mnist_load_dataset("mnist", size=args.size, keep=True)
    else:
        _, train_set, test_set = load_dataset("%s/%s" % (args.data_dir, args.dataset), valid=False, dataset_seed=args.seed)
    wlog("N_train: {}".format(train_set.size))
    wlog("train_l sum {}".format(train_set.data.sum()))
    summary(train_set.data)

    args.num_classes = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'cifar100': 100}[args.dataset]
    test_set = data_util.TensorDataset(torch.FloatTensor(test_set.data), torch.LongTensor(test_set.label))
    test_loader = data_util.DataLoader(test_set, 128, False)
    return train_set, test_loader


def init_model(args):
    arch = getattr(models, args.arch)
    model = arch(args)
    if args.debug:
        # weights init is based on numpy, so only need np.random.seed()
        np.random.seed(args.seed)
        model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        exp_marker = "L0VAT-sup/%s" % args.dataset
        checkpoint = load_checkpoint_by_marker(args, exp_marker)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
    if args.dataset == "mnist":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    else:
        scheduler = "linear"
    model = model.to(args.device)
    model.train()
    return model, optimizer, scheduler, start_epoch


def train(dataset_kit, model_kit, args):
    train_set, test_loader = dataset_kit
    model, optimizer, scheduler, start_epoch = model_kit

    # Define losses.
    criterion = nn.CrossEntropyLoss()
    reg_component = VAT(args)
    if 'at' == args.trainer:
        reg_component = AT(args)

    # show the masked images and masks
    idx = []
    if args.vis and "fig" in args.log_arg:
        np.random.seed(0)
        for i in range(10):
            ind = np.where(train_set.label == i)[0]
            idx.append(np.random.choice(ind, 1))
        idx = np.concatenate(idx)
        wlog("select index images %s" % str(list(idx)))
        if "vat" in args.trainer:
            show_and_save_vat_generated_demo(reg_component, model, train_set.data[idx], args, 0, 0.1, 1)
        elif "at" in args.trainer:
            show_and_save_at_generated_demo(reg_component, model, train_set.data[idx], train_set.label[idx], args, 0, 0.1, 1)

    # train
    start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        for it in range(args.num_batch_it):

            x, t = train_set.get(args.batch_size, aug_trans=args.aug_trans, aug_flip=args.aug_flip)
            images = torch.FloatTensor(x).to(args.device)
            labels = torch.LongTensor(t).to(args.device)

            logits = model(images)

            # supervised loss
            ce_loss = criterion(logits, labels)
            sup_loss = ce_loss

            ul_loss = 0
            if "ce" == args.trainer:
                total_loss = sup_loss
            elif "at" == args.trainer:
                ul_loss, d = reg_component(model, images, labels)
                total_loss = sup_loss + ul_loss
            elif "vat" in args.trainer:
                ul_loss = reg_component(model, images, kl_way=args.kl)
                total_loss = sup_loss + ul_loss
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if ((epoch % args.log_interval) == 0 and it == args.num_batch_it - 1) or (args.debug and it < 5 and epoch == 0):
                n_err, test_loss = evaluate_classifier(model, test_loader, args.device)
                acc = 1 - n_err / len(test_loader.dataset)
                cost_time = time.time() - start_time
                wlog("Epoch: %d Train Loss %.4f ce %.5f, ul loss %.5f, test loss %.5f, test acc %.4f, time %.2f" % (epoch, total_loss, ce_loss, ul_loss, test_loss, acc, cost_time))

                if args.vis and it == args.num_batch_it - 1:
                    if (epoch + 1) * 10 % args.num_epochs == 0:
                        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': test_loss, 'acc': acc},
                                   "%s/model.pth" % args.dir_path)
                        if epoch > args.num_epochs / 2:
                            shutil.copy("%s/model.pth" % args.dir_path, "%s/model_%d.pth" % (args.dir_path, epoch+1))
                    pred_y = torch.max(logits, dim=1)[1]
                    train_acc = 1.0 * torch.sum(pred_y == labels).item() / pred_y.shape[0]

                    dicts = {"Train/CELoss": ce_loss, "Train/UnsupLoss": ul_loss, "Train/Loss": total_loss, "Test/Acc": acc, "Test/Loss": test_loss, "Train/Acc": train_acc}
                    vis_step(args.writer, epoch, dicts)

                    save_interval = 20
                    if epoch % save_interval == 0 and "fig" in args.log_arg:
                        if "vat" in args.trainer:
                            show_and_save_vat_generated_demo(reg_component, model, train_set.data[idx], args, epoch+1, acc, save_interval)
                        elif "at" in args.trainer:
                            show_and_save_at_generated_demo(reg_component, model, train_set.data[idx], train_set.label[idx], args, epoch+1, acc, save_interval)
                start_time = time.time()

        if scheduler == "linear":
            lr = adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]
        if epoch % args.log_interval == 0:
            wlog("learning rate %f" % lr)
            if args.vis:
                args.writer.add_scalar("Optimizer/LearningRate", lr, epoch)


def main(args):
    """Training function."""
    set_framework_seed(args.seed, args.debug)
    dataset_kit = get_data(args)
    model_kit = init_model(args)
    train(dataset_kit, model_kit, args)


if __name__ == "__main__":
    arg = parse_args()
    # noinspection PyBroadException
    try:
        main(arg)
    except KeyboardInterrupt:
        if arg.dir_path:
            os.rename(arg.dir_path, arg.dir_path + "_stop")
    except BaseException as err:

        traceback.print_exc()
        if arg.dir_path:
            shutil.rmtree(arg.dir_path)
