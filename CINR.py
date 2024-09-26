import math
import torch
from torch import nn
import torch.nn.functional as F


# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# tanh activation with scale

class Tanh(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.tanh(self.w0 * x)


# sine activation with scale

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer

class Siren(nn.Module):
    def __init__(
        self, dim_in, dim_out, w0 = 1., c = 6., dropout = 0., 
            is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


# siren network

class SirenNet(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., dropout = 0.,
            use_bias = True, final_activation = None):
        super().__init__()
        #self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        #self.dim_out = dim_out
        self.num_layers = num_layers
        #self.w0_initial = w0_initial

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            #layer_w0 = self.w0_initial if is_first else self.w0
            #layer_dim_in = self.dim_in if is_first else self.dim_hidden
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = False, activation = final_activation)

    def forward(self, x, mods):

        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if len(mod.shape) < len(x.shape):
                x = x * mod.unsqueeze(1)
            else:
                x = x * mod

        return self.last_layer(x)


class SirenSeg(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, coor_dim, seg_dim, num_layers, w0 = 1., w0_initial = 30., dropout = 0.,
            use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        self.coor_layer = Siren(dim_in=dim_hidden, dim_out=coor_dim, w0=w0, use_bias=False, activation=nn.Tanh())
        self.seg_layer = Siren(dim_in=dim_hidden, dim_out=seg_dim, w0=w0, use_bias=False, activation=nn.Identity())

    def forward(self, x, mods):

        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if len(mod.shape) < len(x.shape):
                x = x * mod.unsqueeze(1)
            else:
                x = x * mod

        return self.coor_layer(x), self.seg_layer(x)


class SegmentWrapper(nn.Module):
    def __init__(self, net, num_classes, cpc, seg_classes, condition_dim, num_layers=1):
        super().__init__()
        if isinstance(net, SirenNet):
            self.task = 'cls'
        elif isinstance(net, SirenSeg):
            self.task = 'seg'
        else:
            print('SirenWrapper must receive a Siren network')
            return

        self.net = net
        self.class_lookup = nn.Embedding(num_classes * cpc, condition_dim)
        #self.seg_lookup = nn.Embedding(seg_classes, condition_dim)
        self.num_layers = num_layers

        self.layers = nn.ModuleList([])

        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.ReLU()))

    def forward(self, noise, indices):
        # noise:  [B, N, 1]
        # latent: [B, C]
        # output: [B, N, 3]

        mod = torch.relu(self.class_lookup(indices))
        mods = [mod]

        for layer in self.layers:
            mod = layer(mod)
            mods.append(mod)

        return self.net(noise, tuple(mods))


# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, num_classes, cpc, condition_dim, num_layers=1):
        super().__init__()
        if isinstance(net, SirenNet):
            self.task = 'cls'
        elif isinstance(net, SirenSeg):
            self.task = 'seg'
        else:
            print('SirenWrapper must receive a Siren network')
            return

        self.net = net
        self.lookup = nn.Embedding(num_classes * cpc, condition_dim)
        self.num_layers = num_layers

        self.layers = nn.ModuleList([])

        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.ReLU()))

    def forward(self, noise, indices):

        mod = torch.relu(self.lookup(indices))
        mods = [mod]

        for layer in self.layers:
            mod = layer(mod)
            mods.append(mod)

        return self.net(noise, tuple(mods))


class InvariantWrapper(nn.Module):
    def __init__(self, net, hidden_dim=32, w0=6.):
        super().__init__()

        self.net = net
        self.w0 = w0
        #self.sign_encoder = Siren(1, hidden_dim, w0=w0, is_first=True,  use_bias=False)
        #self.sign_decoder = Siren(hidden_dim, 1, w0=1., is_first=False, use_bias=False, activation=nn.Tanh())

    def forward(self, ptn):
        # [B, N, 3] * [B, 3, 3] -> [B, N, 3]
        # ptn = torch.bmm(ptn, vec)

        '''
        ptn = ptn.unsqueeze(-1)

        feat = self.sign_encoder(ptn).mean(dim=1, keepdim=True)  # [B, N, 3, d] -> [B, 1, 3, d]
        feat = self.sign_decoder(feat)                           # [B, 1, 3, d] -> [B, 1, 3, 1]
        sign = torch.sign(feat)

        ptn = ptn * sign         # [B, N, 3, 1] * [B, 1, 3, 1] -> [B, N, 3, 1]
        ptn = ptn.squeeze(-1)
        '''

        sign = torch.sign(torch.sin(self.w0 * ptn).mean(dim=1, keepdim=True))
        ptn = ptn * sign

        return self.net(ptn)

    def canonical(self, ptn):
        '''
        ptn = ptn.unsqueeze(-1)

        feat = self.sign_encoder(ptn).mean(dim=1, keepdim=True)  # [B, N, 3, d] -> [B, 1, 3, d]
        feat = self.sign_decoder(feat)                           # [B, 1, 3, d] -> [B, 1, 3, 1]
        sign = torch.sign(feat)

        ptn = ptn * sign         # [B, N, 3, 1] * [B, 1, 3, 1] -> [B, N, 3, 1]
        ptn = ptn.squeeze(-1)
        '''

        sign = torch.sign(torch.sin(self.w0 * ptn).mean(dim=1, keepdim=True))
        ptn = ptn * sign

        return ptn

    def embed(self, ptn):
        ptn = ptn.unsqueeze(-1)

        feat = self.sign_encoder(ptn).mean(dim=1, keepdim=True)  # [B, N, 3, d] -> [B, 1, 3, d]
        feat = self.sign_decoder(feat)                           # [B, 1, 3, d] -> [B, 1, 3, 1]
        sign = torch.sign(feat)

        ptn = ptn * sign         # [B, N, 3, 1] * [B, 1, 3, 1] -> [B, N, 3, 1]
        ptn = ptn.squeeze(-1)

        return self.net.embed(ptn)

