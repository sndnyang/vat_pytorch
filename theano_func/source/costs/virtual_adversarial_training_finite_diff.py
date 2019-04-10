import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

from .adversarial_training import get_main_obj, get_normalized_vector, get_perturbation
from .virtual_adversarial_training import get_kl


def error(pred, t):
    # print("costs/error")
    return T.sum(T.neq(T.argmax(pred, axis=1), T.argmax(t, axis=1)))


def virtual_adversarial_training_finite_diff(x, t, forward_func,
                                             main_obj_type,
                                             epsilon,
                                             lamb=numpy.asarray(1.0, theano.config.floatX),
                                             norm_constraint='L2',
                                             num_power_iter=1,
                                             unchain_y=True,
                                             xi=1e-6,
                                             x_for_generating_adversarial_examples=None,
                                             forward_func_for_generating_adversarial_examples=None, ):
    ret = 0
    y = forward_func(x)
    sup_loss = get_main_obj(y, t, main_obj_type)
    ret += sup_loss

    if x_for_generating_adversarial_examples is not None:
        x = x_for_generating_adversarial_examples
        y = forward_func(x)
    if forward_func_for_generating_adversarial_examples is not None:
        forward_func = forward_func_for_generating_adversarial_examples
        y = forward_func(x)
    rng = RandomStreams(seed=numpy.random.randint(1234))
    d = rng.normal(size=x.shape, dtype=theano.config.floatX)
    # power_iteration
    for power_iter in range(num_power_iter):
        d = xi * get_normalized_vector(d)
        y_d = forward_func(x + d)
        Hd = T.grad(get_kl(y_d, y, main_obj_type).mean(), wrt=d) / xi
        Hd = theano.gradient.disconnected_grad(Hd)
        d = Hd
    r_vadv = get_perturbation(d, epsilon, norm_constraint)
    if unchain_y:
        y_hat = theano.gradient.disconnected_grad(y)
        pred = forward_func(x + r_vadv)
        vadv_cost = get_kl(pred, y_hat, main_obj_type).mean()
        err = error(pred, y_hat)
    else:
        pred = forward_func(x + r_vadv)
        vadv_cost = get_kl(pred, y, main_obj_type, include_ent_term=True).mean()
        err = error(pred, y)
    unsup_loss = vadv_cost
    ret += lamb * unsup_loss

    return ret, sup_loss, unsup_loss, err


def LDS_finite_diff(x,
                    forward_func,
                    main_obj_type,
                    epsilon,
                    norm_constraint='L2',
                    num_power_iter=1,
                    xi=1e-6):
    rng = RandomStreams(seed=numpy.random.randint(1234))
    y = forward_func(x)
    d = rng.normal(size=x.shape, dtype=theano.config.floatX)
    # power_iteration
    for power_iter in range(num_power_iter):
        d = xi * get_normalized_vector(d)
        y_d = forward_func(x + d)
        Hd = T.grad(get_kl(y_d, y, main_obj_type).mean(), wrt=d) / xi
        Hd = theano.gradient.disconnected_grad(Hd)
        d = Hd
    r_vadv = get_perturbation(d, epsilon, norm_constraint)
    return -get_kl(forward_func(x + r_vadv), y, main_obj_type, include_ent_term=True)


def average_LDS_finite_diff(x,
                            forward_func,
                            main_obj_type,
                            epsilon,
                            norm_constraint='L2',
                            num_power_iter=1):
    return LDS_finite_diff(x=x,
                           forward_func=forward_func,
                           main_obj_type=main_obj_type,
                           epsilon=epsilon,
                           norm_constraint=norm_constraint,
                           num_power_iter=num_power_iter
                           ).mean()
