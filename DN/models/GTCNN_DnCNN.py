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


class GTCNN_DnCNN(nn.Module):
    def __init__(self, NetConfig):
        super(GTCNN_DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        self.NetConfig = NetConfig
        self.weight_list=[]
        self.depth = NetConfig['depth']
        self.n_channels = NetConfig['n_channels']
        self.Selectmodel = NetConfig['Selectmodel']
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

        self.DnCNN = nn.ModuleList()
        for _ in range(0):
            op = OperationLayer(self.n_channels,use_bnorm=NetConfig['use_bnorm'])
            self.DnCNN += [op]

        self.out=nn.Sequential(nn.Conv2d(in_channels=self.n_channels, out_channels=NetConfig['image_channels'], kernel_size=NetConfig['kernel_size'], padding=padding, bias=False))
    def forward(self, x):
        y = x
        x = self.input(x)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GTL) or not isinstance(layer, OperationLayer) :
                # pass
                if self.Selectmodel == 'GTL':
                    x = layer(x)
                # x = layer(x)
            else:
                if self.Selectmodel == 'CBR':
                     x = layer(x)
                # x = layer(x,weights=None)
                # x = layer(x,weights)
                # x=x*weights
        for i, layer in enumerate(self.DnCNN):
            x = layer(x)
        out = y-self.out(x)
        # out = self.out(x)
        return out




