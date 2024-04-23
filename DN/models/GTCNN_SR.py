#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .operations import *
import numpy as np
from .GTL import GTL


## Operation layer
class OperationLayer(nn.Module):
    def __init__(self, C,use_bnorm=True):
        super(OperationLayer, self).__init__()
        self.C_out = C
        self.C_in = C
        self.op = Conv(self.C_in, self.C_out,use_bnorm)
    def forward(self, x, weights=None):
        if weights is not None:
            return self.op(x)*weights
        else:
            return self.op(x)


class GTCNN_SR(nn.Module):
    def __init__(self, NetConfig):
        super(GTCNN_SR, self).__init__()
        kernel_size = 3
        padding = 1
        self.NetConfig = NetConfig
        self.weight_list=[]
        self.depth = NetConfig['depth']
        self.n_channels = NetConfig['n_channels']
        scale = NetConfig['scale']
        rgb_range = 255
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.layers = nn.ModuleList()
        self.input = nn.Sequential(nn.Conv2d(in_channels=NetConfig['image_channels'], out_channels=self.n_channels, kernel_size=NetConfig['kernel_size'], padding=padding, bias=True),
                                    nn.ReLU(inplace=True),)
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            atn = GTL(netconf = NetConfig ,GTL_IC=NetConfig['GTL_IC'],GTL_OC=NetConfig['GTL_OC'],n_chan=NetConfig['GTL_NC'],
                num_cbr=NetConfig['GTL_num_cbr'],act=NetConfig['GTL_ACT'],pooling=NetConfig['GTL_pooling'],upmodule=NetConfig['GTL_upmodule'],stage=NetConfig['GTL_stages'],use_bnorm=NetConfig['use_bnorm'],stage_option=NetConfig['GTL_stage_option'],concat_type=NetConfig['GTL_concat_type'])
            self.layers += [atn]

            op = OperationLayer(self.n_channels,use_bnorm=NetConfig['use_bnorm'])
            self.layers += [op]

        self.out=nn.Sequential(Upsampler(default_conv, scale, n_feat=self.n_channels, act=False),nn.Conv2d(in_channels=self.n_channels, out_channels=NetConfig['image_channels'], kernel_size=NetConfig['kernel_size'], padding=padding, bias=False))
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.input(x)
        y = x

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GTL) or not isinstance(layer, OperationLayer) :
                weights = layer(x)
            else:
                x = layer(x,weights)
        # out = y-self.out(x)
        out = self.out(x+y)
        # out = self.out(x+y)
        out = self.add_mean(out)
        return out



