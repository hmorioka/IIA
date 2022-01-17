"""IIA-GCL model"""


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
class NetGaussScaleMean(nn.Module):
    def __init__(self, h_sizes, num_dim, num_data, num_basis, ar_order=1, h_sizes_z=None, pool_size=2):
        """ Network model for gaussian distribution with scale-mean modulations
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_dim: number of dimension
             num_data: number of data points
             num_basis: number of fourier bases
             ar_order: model order of AR
             h_sizes_z: number of channels for each layer of MLP_z [num_layer+1] (first size is input-dim)
             pool_size: pool size of max-out nonlinearity
         """
        super(NetGaussScaleMean, self).__init__()
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
        # Q
        self.wr1 = nn.Linear(2*num_basis, h_sizes[-1]+h_sizes_z[-1], bias=True)
        self.wr2 = nn.Linear(2*num_basis, h_sizes[-1]+h_sizes_z[-1], bias=True)
        self.a = nn.Linear(1, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)
        self.c = nn.Linear(1, 1, bias=False)
        self.d = nn.Linear(1, 1, bias=False)
        self.e = nn.Linear(1, 1, bias=False)
        self.f = nn.Linear(1, 1, bias=False)
        self.g = nn.Linear(1, 1, bias=False)
        self.m = nn.Linear(1, 1, bias=False)
        self.num_dim = num_dim
        self.num_data = num_data
        self.num_basis = num_basis
        self.ar_order = ar_order

        # initialize
        for k in range(len(self.layer)):
            torch.nn.init.xavier_uniform_(self.layer[k].weight)
        for k in range(len(self.layerz)):
            torch.nn.init.xavier_uniform_(self.layerz[k].weight)
        torch.nn.init.xavier_uniform_(self.wr1.weight)
        torch.nn.init.xavier_uniform_(self.wr2.weight)
        torch.nn.init.constant_(self.a.weight, 1)
        torch.nn.init.constant_(self.b.weight, 1)
        torch.nn.init.constant_(self.c.weight, 1)
        torch.nn.init.constant_(self.d.weight, 1)
        torch.nn.init.constant_(self.e.weight, 1)
        torch.nn.init.constant_(self.f.weight, 1)
        torch.nn.init.constant_(self.g.weight, 1)
        torch.nn.init.constant_(self.m.weight, 0)

    def forward(self, x, t):
        """ forward
         Args:
             x: input [batch, time(t:t-p), dim]
             t: time index [batch, dim]
         """
        batch_size = x.size()[0]
        xz = x[:, 1:, :]

        # concatenate t and t*
        t_shfl = t[torch.randperm(batch_size)]
        t = torch.cat([t, t_shfl], dim=0)

        # h
        h = x.reshape([batch_size, -1])
        for k in range(len(self.layer)):
            h = self.layer[k](h)
            if k != len(self.layer)-1:
                h = self.maxout(h)
        h = torch.cat([h, h], dim=0)

        # hz
        hz = xz.reshape([batch_size, -1])
        for k in range(len(self.layerz)):
            hz = self.layerz[k](hz)
            if k != len(self.layerz)-1:
                hz = self.maxout(hz)
        hz = torch.cat([hz, hz], dim=0)

        # Q
        # fourier bases
        fn_basis = 2 * np.pi * torch.arange(1, self.num_basis+1, device=x.device).reshape([1, -1]) * t.type(torch.float).reshape([-1, 1]) / self.num_data
        t_basis = torch.cat([torch.sin(fn_basis), torch.cos(fn_basis)], dim=1)
        # modulation
        t_mod_log1 = self.wr1(t_basis)
        t_mod1 = torch.exp(t_mod_log1)
        t_mod2 = self.wr2(t_basis)

        h_square = h**2
        hz_square = hz**2

        h_mod = h_square * t_mod1[:, 0:h.size()[1]] * self.a.weight + h * t_mod1[:, 0:h.size()[1]] * t_mod2[:, 0:h.size()[1]] * self.b.weight
        hz_mod = hz_square * t_mod1[:, h.size()[1]:] * self.c.weight + hz * t_mod1[:, h.size()[1]:] * t_mod2[:, h.size()[1]:] * self.d.weight

        Q = torch.mean(h_mod, dim=[1]) + torch.mean(hz_mod, dim=[1])
        Qbar = torch.mean(h_square * self.e.weight, dim=[1]) + torch.mean(hz_square * self.f.weight, dim=[1])
        Z = torch.mean(t_mod_log1 * self.g.weight, dim=[1])
        logits = - Q + Qbar + Z + self.m.weight[0]

        return logits, h, hz, t_mod1, t_mod2
