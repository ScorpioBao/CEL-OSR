import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

def uncertainty_loss(logits, num_classes):
    evidence = softplus_evidence(logits)
    alpha = evidence + 1
    S = alpha.sum(dim=1)
    U = num_classes/S
    return U.mean()
