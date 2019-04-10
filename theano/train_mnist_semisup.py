# I don't like docopt... I still prefer argparse
import os
import sys
import traceback

import numpy as np
import theano
import theano.tensor as t_func
from theano.tensor.shared_randomstreams import RandomStreams

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from ExpUtils import *
from CostFunc import get_cost_type_semi
from CostFunc import get_cost_type
from source import optimizers
from source import costs
from models.fnn_mnist import MLP
from collections import OrderedDict
import load_data

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='VAT theano', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--size', type=int, default=100, help='size of training data set 0=all(default: 100)')
    parser.add_argument('--num-valid', type=int, default=1000, help='size of validation set (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='N', help='number of epochs (default: 1000)')
    parser.add_argument('--num-batch-it', type=int, default=500, metavar='N', help='number of batch iterations (default: 500)')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--gpu-id', type=str, default="5", metavar='N', help='gpu id list (default: 5)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='iterations to wait before logging status')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training data set (default: 100)')
    parser.add_argument('--ul-batch-size', type=int, default=250, help='size of training data set 0=all(default: 100)')
    parser.add_argument('--layer-sizes', type=str, default='784-1200-1200-10', help='784-1200-1200-10 MLP for semi supervised learning on mnist')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.002)')
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for VAT (default: 0.3)')
    parser.add_argument('--xi', type=float, default=1e-6, help='xi for VAT (default: 1e-6)')
    parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--affine', action='store_true', default=False, help='batch norm use affine')
    parser.add_argument('--top-bn', action='store_true', default=False, help='use top batch norm')
    parser.add_argument('--aug-trans', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--aug-flip', action='store_true', default=False, help='data augmentation flip')
    parser.add_argument('--entmin', action='store_true', default=False, help='visual by tensor board')
    parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory')
    parser.add_argument('--log-arg', type=str, default='eps-xi', metavar='S', help='show the arguments in directory name')
    parser.add_argument('-r', '--resume', type=str, default='', metavar='S', help='resume from pth file')
    parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')

    args = parser.parse_args()
    args.dir_path = None

    if int(arg.gpu_id) != "-1":
        os.environ['THEANO_FLAGS'] = "device=cuda%s,floatX=float32" % arg.gpu_id

    if args.debug:
        args.num_batch_it = 1
        args.vis = False
        args.log_arg += "-debug"
        args.log_interval = 1
        args.num_epochs = 50

    if args.vis:
        args_dict = vars(args)
        run_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        exp_marker = "-".join("%s=%s" % (e, str(args_dict.get(e, "None"))) for e in args.log_arg.split("-"))
        exp_marker = "VAT-theano-semi/mnist/%s_%s" % (exp_marker, run_time)
        base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
        dir_path = os.path.join(base_dir, exp_marker)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        info = '\n'.join(str(e) for e in sorted(vars(args).items()))
        args.dir_path = dir_path
        args.writer = SummaryWriter(log_dir=dir_path)
    wlog("args in this experiment %s", '\n'.join(str(e) for e in sorted(vars(args).items())))
    return args, kwargs

def train(args):
    wlog(args)
    np.random.seed(args.seed)

    dataset = load_data.load_mnist_for_semi_sup(n_l=args.num_labeled_samples,
                                                n_v=args.num_validation_samples)

    x_train, t_train, ul_x_train = dataset[0]
    x_test, t_test = dataset[2]

    np.random.seed(args.seed)
    layer_sizes = [int(layer_size) for layer_size in args.layer_sizes.split('-')]
    model = FNN_MNIST(layer_sizes=layer_sizes, top_bn=args.top_bn)

    x = t_func.matrix()
    ul_x = t_func.matrix()
    t = t_func.ivector()

    cost_semi, sup, unsup, p_err = get_cost_type_semi(model, x, t, ul_x, args)
    nll = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_test)
    error = costs.error(x=x, t=t, forward_func=model.forward_test)

    optimizer = optimizers.ADAM(cost=cost_semi, params=model.params, alpha=args.lr)

    index = t_func.iscalar()
    ul_index = t_func.iscalar()
    batch_size = args.batch_size
    ul_batch_size = args.ul_batch_size

    f_train = theano.function(inputs=[index, ul_index], outputs=[cost_semi, sup, unsup, p_err, ul_x], updates=optimizer.updates,
                              givens={
                                  x: x_train[batch_size * index:batch_size * (index + 1)],
                                  t: t_train[batch_size * index:batch_size * (index + 1)],
                                  ul_x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]},
                              on_unused_input='ignore')
    f_nll_train = theano.function(inputs=[index], outputs=nll,
                                  givens={
                                      x: x_train[batch_size * index:batch_size * (index + 1)],
                                      t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_nll_test = theano.function(inputs=[index], outputs=nll,
                                 givens={
                                     x: x_test[batch_size * index:batch_size * (index + 1)],
                                     t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_error_train = theano.function(inputs=[index], outputs=error,
                                    givens={
                                        x: x_train[batch_size * index:batch_size * (index + 1)],
                                        t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_error_test = theano.function(inputs=[index], outputs=error,
                                   givens={
                                       x: x_test[batch_size * index:batch_size * (index + 1)],
                                       t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_lr_decay = theano.function(inputs=[], outputs=optimizer.alpha,
                                 updates={optimizer.alpha: theano.shared(
                                     np.array(args.learning_rate_decay).astype(
                                         theano.config.floatX)) * optimizer.alpha})

    # Shuffle training set
    randix = RandomStreams(seed=np.random.randint(1234)).permutation(n=x_train.shape[0])
    update_permutation = OrderedDict()
    update_permutation[x_train] = x_train[randix]
    update_permutation[t_train] = t_train[randix]
    f_permute_train_set = theano.function(inputs=[], outputs=x_train, updates=update_permutation)

    # Shuffle unlabeled training set
    ul_randix = RandomStreams(seed=np.random.randint(1234)).permutation(n=ul_x_train.shape[0])
    update_ul_permutation = OrderedDict()
    update_ul_permutation[ul_x_train] = ul_x_train[ul_randix]
    f_permute_ul_train_set = theano.function(inputs=[], outputs=ul_x_train, updates=update_ul_permutation)

    statuses = {'nll_train': [], 'error_train': [], 'nll_test': [], 'error_test': []}

    n_train = x_train.get_value().shape[0]
    n_test = x_test.get_value().shape[0]
    n_ul_train = ul_x_train.get_value().shape[0]

    l_i = 0
    ul_i = 0
    for epoch in range(int(args.num_epochs)):

        # since theano uses its now random permutation, hard to get the same permutation
        # f_permute_train_set()
        # f_permute_ul_train_set()
        for it in range(int(args.num_batch_it)):
            loss, sup, unsup, pert_err, ul_x = f_train(l_i, ul_i)
            l_i = 0 if l_i >= n_train / batch_size - 1 else l_i + 1
            ul_i = 0 if ul_i >= n_ul_train / ul_batch_size - 1 else ul_i + 1
            if it % 100 == 0:
                print("iteration %d" % it)
                print("total loss %g" % loss)
                print("sup %g" % sup)
                print("unsup %g" % unsup)
                # print("pert err %d" % pert_err)
                print("unlabeled data %g" % ul_x.sum())
                args.writer.add_scalar("Train/total_loss", loss, epoch * args.num_batch_it + it)
                args.writer.add_scalar("Train/iter_xent_loss", sup, epoch * args.num_batch_it + it)
                args.writer.add_scalar("Train/xent_loss", sup, epoch)
                args.writer.add_scalar("Train/iter_unsup_loss", unsup, epoch * args.num_batch_it + it)
                args.writer.add_scalar("Train/unsup_loss", unsup, epoch)
                args.writer.add_scalar("Train/un_kl_loss", unsup, epoch * args.num_batch_it + it)
                args.writer.add_scalar("Train/pert_err", pert_err, epoch * args.num_batch_it + it)
                args.writer.add_scalar("Train/pert_err_rate", 1.0 * pert_err / ul_batch_size, epoch * args.num_batch_it + it)
                if i % 100 == 0:
                    # it takes time to compute
                    sum_error_test = np.sum(np.array([f_error_test(i) for i in range(n_test // batch_size)]))
                    acc = 1 - 1.0*sum_error_test/n_test
                    args.writer.add_scalar("Test/iter_acc", acc, epoch * int(args.num_batch_it) + it)

        # usually I don't compute this
        # sum_nll_train = np.sum(np.array([f_nll_train(i) for i in range(n_train // batch_size)])) * batch_size
        # sum_error_train = np.sum(np.array([f_error_train(i) for i in range(n_train // batch_size)]))
        # statuses['nll_train'].append(sum_nll_train / n_train)
        # statuses['error_train'].append(sum_error_train)
        sum_nll_test = np.sum(np.array([f_nll_test(i) for i in range(n_test // batch_size)])) * batch_size
        sum_error_test = np.sum(np.array([f_error_test(i) for i in range(n_test // batch_size)]))
        statuses['nll_test'].append(sum_nll_test / n_test)
        statuses['error_test'].append(sum_error_test)
        wlog("[Epoch] %d" % epoch)
        acc = 1 - 1.0 * statuses['error_test'][-1] / n_test
        wlog("nll_test : %f error_test : %d accuracy:%f" % (statuses['nll_test'][-1], statuses['error_test'][-1], acc))
        args.writer.add_scalar("Train/Loss", statuses['nll_train'][-1], epoch * args.num_batch_it)
        args.writer.add_scalar("Train/Error", statuses['error_train'][-1], epoch * args.num_batch_it)
        args.writer.add_scalar("Test/Loss", statuses['nll_test'][-1], epoch * args.num_batch_it)
        args.writer.add_scalar("Test/Acc", acc, epoch * args.num_batch_it)
        f_lr_decay()
    # fine_tune batch stat
    f_fine_tune = theano.function(inputs=[ul_index], outputs=model.forward_for_finetuning_batch_stat(x),
                                  givens={x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]})
    [f_fine_tune(i) for i in range(n_ul_train // ul_batch_size)]

    sum_nll_test = np.sum(np.array([f_nll_test(i) for i in range(n_test // batch_size)])) * batch_size
    sum_error_test = np.sum(np.array([f_error_test(i) for i in range(n_test // batch_size)]))
    statuses['nll_test'].append(sum_nll_test / n_test)
    statuses['error_test'].append(sum_error_test)
    acc = 1 - 1.0 * statuses['error_test'][-1] / n_test)
    wlog("final nll_test: %f error_test: %d accuracy:%f" % (statuses['nll_test'][-1], statuses['error_test'][-1], acc)
    args.writer.add_scalar("Test/Loss", statuses['nll_test'][-1], epoch * args.num_batch_it)
    args.writer.add_scalar("Test/Acc", acc, epoch * args.num_batch_it)

    make_sure_path_exists("./trained_model")
    pickle.dump((model, statuses, args), open('./trained_model/' + args.save_filename, 'wb'),
                pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    arg.exp = "1"
    arg.dataset = "mnist"
    saver = ExpSaver("VAT_theano_semi-%s" % arg.cost_type, arg, arg.log_arg.split("-"), None)
    # noinspection PyBroadException
    try:
        train(arg)
    except BaseException as err:
        traceback.print_exc()
        saver.delete_dir()
        sys.exit(-1)
    saver.finish_exp({"num_epochs": 10})
