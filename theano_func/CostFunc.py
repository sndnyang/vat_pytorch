import sys
from .source import costs


def get_cost_type_semi(model, x, t, ul_x, args):
    if args.trainer == 'MLE':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train)
        cost = [cost, cost, cost, cost]
    elif args.trainer == 'L2':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params, coeff=float(args.lamb))
    elif args.trainer == 'AT':
        cost = costs.adversarial_training(x, t, model.forward_train,
                                          'CE',
                                          epsilon=float(args.eps),
                                          lamb=float(args.lamb),
                                          forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif args.trainer == 'VAT':
        cost = costs.virtual_adversarial_training(x, t, model.forward_train,
                                                  'CE',
                                                  epsilon=float(args.eps),
                                                  num_power_iter=int(args.num_power_iter),
                                                  x_for_generating_adversarial_examples=ul_x,
                                                  forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif args.trainer == 'VAT_f':
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              xi=1e-6,
                                                              epsilon=float(args.eps),
                                                              num_power_iter=int(args.num_power_iter),
                                                              x_for_generating_adversarial_examples=ul_x,
                                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    else:
        print("method not exists")
        sys.exit(-1)
    return cost


def get_cost_type(model, x, t, args):
    forward_func = None
    if args.dataset == "mnist":
        forward_func = model.forward_no_update_batch_stat
    if args.trainer == 'MLE' or args.trainer == 'dropout':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train)
    elif args.trainer == 'L2':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params, coeff=float(args.lamb))
    elif args.trainer == 'AT':
        cost = costs.adversarial_training(x, t, model.forward_train,
                                          'CE',
                                          epsilon=float(args.eps),
                                          lamb=float(args.lamb),
                                          forward_func_for_generating_adversarial_examples=forward_func)
    elif args.trainer == 'VAT':
        cost = costs.virtual_adversarial_training(x, t, model.forward_train,
                                                  'CE',
                                                  epsilon=float(args.eps),
                                                  num_power_iter=int(args.num_power_iter),
                                                  forward_func_for_generating_adversarial_examples=forward_func)
    elif args.trainer == 'VAT_f':
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              epsilon=float(args.eps),
                                                              num_power_iter=int(args.num_power_iter),
                                                              forward_func_for_generating_adversarial_examples=forward_func)
    else:
        print("method not exists")
        sys.exit(-1)
    return cost
