"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import numpy as np
import torch


def _l2_dist(X):
    X = X.view(len(X), -1)
    XX = X @ X.t()
    X_sqnorms = torch.diag(XX)
    X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
    return X_L2.clone().detach().cpu().numpy().reshape(-1).tolist()


def median_distance(f_net, g_net, train_loader, sigma_update_sampling_rate, func='median', device='cuda'):
    if func not in {'mean', 'median'}:
        raise ValueError(func)

    f_net.train()
    g_net.train()

    f_dist, g_dist = [], []
    for idx, (x, _) in enumerate(train_loader):
        if idx > len(train_loader) * sigma_update_sampling_rate:
            break
        x = x.to(device)
        n_data = len(x)

        _, f_feats = f_net(x)
        f_feats = f_feats.view(n_data, -1)
        _f_dist = _l2_dist(f_feats)
        f_dist.extend(_f_dist)


        _, g_feats = g_net(x)
        g_feats = g_feats.view(n_data, -1)
        _g_dist = _l2_dist(g_feats)
        g_dist.extend(_g_dist)

    f_dist, g_dist = np.array(f_dist), np.array(g_dist)

    if func == 'median':
        f_sigma, g_sigma = np.median(f_dist), np.median(g_dist)
    else:
        f_sigma, g_sigma = np.mean(f_dist), np.mean(g_dist)

    return np.sqrt(f_sigma), np.sqrt(g_sigma)


def feature_dimension(f_net, g_net, train_loader, device='cuda'):
    f_net.train()
    g_net.train()

    for x, _ in train_loader:
        x = x.to(device)
        n_data = len(x)

        _, f_feats = f_net(x)
        f_feats = f_feats.view(n_data, -1)
        f_dim = f_feats.size()[1]


        _, g_feats = g_net(x)
        g_feats = g_feats.view(n_data, -1)
        g_dim = g_feats.size()[1]
        return np.sqrt(f_dim), np.sqrt(g_dim)
