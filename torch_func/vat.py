import torch
import torch.nn as nn
import torch.nn.functional as nfunc
import numpy as np


def _l2_normalize(d):
    t = d.clone()   # remove from the computing graph
    norm = torch.sqrt(torch.sum(t.view(d.shape[0], -1) ** 2, dim=1))
    if len(t.shape) == 4:
        norm = norm.reshape(-1, 1, 1, 1)
    elif len(t.shape) == 3:
        norm = norm.reshape(-1, 1, 1)
    elif len(t.shape) == 2:
        norm = norm.reshape(-1, 1)
    else:
        raise NotImplementedError
    normed_d = t / (norm + 1e-10)
    return normed_d


def _entropy(logits):
    p = nfunc.softmax(logits, dim=1)
    return torch.mean(torch.sum(p * nfunc.log_softmax(logits, dim=1), dim=1))


class VAT(object):

    def __init__(self, device, eps, xi, k=1, use_ent_min=False, debug=False):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.k = k
        self.debug = debug
        try:
            self.kl_div = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_ent_min = use_ent_min

    def __call__(self, model, image, return_noise=False, detach_way=0):
        logits = model(image, update_batch_stats=False)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        log_prob_x = nfunc.log_softmax(logits.detach(), dim=1)
        # np generator is more controllable than torch.randn(image.size())
        d = np.random.standard_normal(image.size())
        d = _l2_normalize(torch.FloatTensor(d).to(self.device))

        for ip in range(self.k):
            d *= self.xi
            d.requires_grad = True
            t = image.detach()
            x_hat = t + d
            logits_x_hat = model(x_hat, update_batch_stats=False)
            if detach_way == 1:
                prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
                adv_distance = torch.mean(self.kl_div(log_prob_x, prob_x_hat).sum(dim=1))
            else:
                # official theano code compute in this way
                log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
                adv_distance = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
            adv_distance.backward()
            grad_x_hat = d.grad
            d = _l2_normalize(grad_x_hat).to(self.device)

        logits_x_hat = model(image + self.eps * d, update_batch_stats=False)
        if detach_way == 1:
            prob_x_hat = torch.exp(nfunc.log_softmax(logits_x_hat, dim=1))
            lds = torch.mean(self.kl_div(log_prob_x, prob_x_hat).sum(dim=1))
        else:
            # official theano code works in this way
            log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
            lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
        if self.debug:
            print("lds value", lds.item())

        if self.use_ent_min:
            # if detach_way == 1, the KL divergence contains this part.
            lds += _entropy(prob_x)

        if return_noise:
            return lds, image + self.eps * d
        else:
            return lds
