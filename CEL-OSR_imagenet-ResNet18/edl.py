import torch
import torch.nn.functional as F
import numpy as np

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(device=get_device())
    return y[labels]


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(y, alpha, alpha_all, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    alpha_y = (alpha_all * y).sum(dim=1,keepdim=True)
    third_term = -100 * (torch.digamma(alpha_y) - torch.digamma(torch.sum(alpha_all,dim=1,keepdim=True)))
    # third_term = -20 * torch.log(alpha_y/sum_alpha) + (-20 * (1/(2*sum_alpha) - 1/(2 * (alpha_y))))
    kl = first_term + second_term



    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def compute_eavuc(preds,labels,confs,uncertainties):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    confs = confs.detach().cpu().numpy()
    uncertainties = uncertainties.detach().cpu().numpy()
    eavuc = 0
    inds_accurate = np.where(preds == labels)[0]
    eavuc += -np.sum(confs[inds_accurate] * np.log(1 - uncertainties[inds_accurate]))
    inds_inaccurate = np.where(preds != labels)[0]
    eavuc += -np.sum((1 - confs[inds_inaccurate]) * np.log(uncertainties[inds_inaccurate]))
    return eavuc

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    # kl_div = annealing_coef * compute_eavuc(pred,labels,confs,uncertainties)
    kl_div = annealing_coef * kl_divergence(kl_alpha,  num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred = alpha.data.max(1)[1]
    labels = y.data.max(1)[1]
    confs = alpha / S
    uncertainties = 20 / S

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / 50, dtype=torch.float32))
    kl_alpha = (alpha - 1) * (1 - y) + 1
    # kl_div = annealing_coef * compute_eavuc(pred,labels,confs,uncertainties)
    kl_div =  annealing_coef * kl_divergence(y, kl_alpha,alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    # evidence = softplus_evidence(output)
    # evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss