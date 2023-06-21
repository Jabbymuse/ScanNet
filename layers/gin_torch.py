import torch.nn as nn
from dgl.nn.pytorch import GINConv as GINConv3

activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5,
                 activation='relu', residual=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x
class GIN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5, activation='relu'):
        super(GIN2, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.activation = activations[activation]
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv3(update_net, 'sum'))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
            self.layers.append(GINConv3(update_net, 'sum'))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv3(update_net, 'sum'))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(g, x)
        return x


class GINDeepSigns2(nn.Module):
    """ Sign invariant neural network
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
 """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(GINDeepSigns2, self).__init__()
        self.enc = GIN2(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        self.k = k


    def forward(self, g, x):
        x = self.enc(g, x) + self.enc(g, -x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x
"""
import torch as th
import dgl
from random import uniform as ru
Naa = 4
m = 2
K = 2

th.manual_seed(seed=0)
g = dgl.graph(([0,1,2,3], [1,0,3,2]))
M = [[[1 for _ in range(6)] for _ in range(m)] for _ in range(Naa)]
feat = th.FloatTensor(M)
conv = GINDeepSigns2(6,6,6,4,m)
res = conv(g,feat)
res2 = conv(g,-feat)
print(res)
print(res2)
"""