"""IIA-TCL model"""


import torch
import torch.nn as nn
from subfunc.showdata import *


# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:])), dim=2)
        return m


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, h_sizes, num_dim, num_class, ar_order=1, h_sizes_z=None, pool_size=2):
        """ Network model for segment-wise stationary model
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_dim: number of dimension
             num_class: number of classes
             ar_order: model order of AR
             h_sizes_z: number of channels for each layer of MLP_z [num_layer+1] (first size is input-dim)
             pool_size: pool size of max-out nonlinearity
         """
        super(Net, self).__init__()
        if h_sizes_z is None:
            h_sizes_z = h_sizes.copy()
        # h
        h_sizes_aug = [num_dim * (ar_order + 1)] + h_sizes
        layer = [nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k]*pool_size) for k in range(1, len(h_sizes_aug)-1)]
        layer.append(nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1]))
        self.layer = nn.ModuleList(layer)
        # hz
        h_sizes_z_aug = [num_dim * ar_order] + h_sizes_z
        layerz = [nn.Linear(h_sizes_z_aug[k-1], h_sizes_z_aug[k]*pool_size) for k in range(1, len(h_sizes_z_aug)-1)]
        layerz.append(nn.Linear(h_sizes_z_aug[-2], h_sizes_z_aug[-1]))
        self.layerz = nn.ModuleList(layerz)
        self.maxout = Maxout(pool_size)
        self.mlr = nn.Linear((h_sizes[-1] + h_sizes_z[-1])*2, num_class)
        self.num_dim = num_dim
        self.ar_order = ar_order

        # initialize
        for k in range(len(self.layer)):
            torch.nn.init.xavier_uniform_(self.layer[k].weight)
        for k in range(len(self.layerz)):
            torch.nn.init.xavier_uniform_(self.layerz[k].weight)
        torch.nn.init.xavier_uniform_(self.mlr.weight)

    def forward(self, x):
        """ forward
         Args:
             x: input [batch, time(t:t-p), dim]
         """
        batch_size = x.size()[0]
        xz = x[:, 1:, :]

        # h
        h = x.reshape([batch_size, -1])
        for k in range(len(self.layer)):
            h = self.layer[k](h)
            if k != len(self.layer)-1:
                h = self.maxout(h)
        h_nonlin = torch.cat((h**2, h), 1)

        # hz
        hz = xz.reshape([batch_size, -1])
        for k in range(len(self.layerz)):
            hz = self.layerz[k](hz)
            if k != len(self.layerz)-1:
                hz = self.maxout(hz)
        hz_nonlin = torch.cat((hz**2, hz), 1)

        # concatenate
        hhz = torch.cat((h_nonlin, hz_nonlin), 1)
        # MLR
        y = self.mlr(hhz)

        return y, h, hz
