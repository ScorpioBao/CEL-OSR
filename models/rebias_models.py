import torch.nn as nn

class ReBiasModels(object):

    def __init__(self, f_net, g_net):
        self.f_net = f_net
        self.g_net = g_net

    def to(self, device):
        self.f_net.to(device)
        self.g_net.to(device)

    def to_parallel(self, device):
        self.f_net = nn.DataParallel(self.f_net.to(device))

        self.g_net = nn.DataParallel(self.g_net.to(device))

    def load_models(self, state_dict):
        self.f_net.load_state_dict(state_dict['f_net'])

        self.g_net.load_state_dict(state_dict['g_net'])

    def train_f(self):
        self.f_net.train()

    def eval_f(self):
        self.f_net.eval()

    def train_g(self):
        self.g_net.train()

    def eval_g(self):
        self.g_net.eval()

    def train(self):
        self.train_f()
        self.train_g()

    def eval(self):
        self.eval_f()
        self.eval_g()

    def forward(self, x):
        f_pred, f_feat = self.f_net(x)
        g_pred, g_feat = self.g_net(x)

        return f_pred, g_pred, f_feat, g_feat

    def __call__(self, x):
        return self.forward(x)