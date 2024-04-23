#!/usr/bin/env python
# -*- coding: utf-8 -*-
from re import A
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .operations import *
import numpy as np

##### add ####

from typing import Optional, Tuple, Union, List
#from labml_helpers.module import Module

#import pdb
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
################


def double_conv(in_channels, out_channels,num_cbr=2,use_bnorm=True):
    layers = []
    if use_bnorm:
        # layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1,bias=True))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(num_cbr-1):
        if use_bnorm:
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=True))
        layers.append(nn.ReLU(inplace=True))
    return  nn.Sequential(*layers)
 

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, upmodule=True,attention=False,num_cbr=2,use_bnorm=True,concat_type='concat'):
        super().__init__()
        self.upmodule = upmodule
        self.concat_type = concat_type
        if self.concat_type ==  'sum':
            in_channels = in_channels //2

        if upmodule == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_channels, out_channels,num_cbr,use_bnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.concat_type ==  'sum':
            x = x2+x1
        else:
            x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels=64,num_cbr=2,pooling ='maxpool',use_bnorm=True):
        super().__init__()
        if pooling == 'maxpool':
            self.subpixel= nn.MaxPool2d(2)
            self.maxpool_conv = nn.Sequential(
                double_conv(in_channels, out_channels,num_cbr,use_bnorm)
            )


    def forward(self, x):
        return self.maxpool_conv(self.subpixel(x))





##############            add code              ############################



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups: int=32):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        #self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
         h = self.conv1(self.act1(self.norm1(x)))
         #h += self.time_emb(t)[:, :, None, None]
         h = self.conv2(self.act2(self.norm2(h)))
         return h + self.shortcut(x)


         """
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups:int=32,dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
         
         h = self.conv1(self.act1(self.norm1(x)))
         #pdb.set_trace()
         h += self.time_emb(self.time_act(t))[:, :, None, None]
         h = self.conv2(self.dropout(self.act2(self.norm2(h)))) 
         return h + self.shortcut(x)
       
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()

        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

        

class SA_Down(Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        
        self.conv = nn.Conv2d(in_channels, in_channels, (3,3), (2,2), (1,1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x,t)
        x = self.attn(x)
        return self.conv(x)

class TSA_Down(Module):
    def __init__(self, in_channels: int, out_channels: int,time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        
        self.conv = nn.Conv2d(in_channels, in_channels, (3,3), (2,2), (1,1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.conv(x)
        x = self.res(x,t)
        return self.attn(x)

class SA_Up(Module):
    def __init__(self, in_channels: int, out_channels: int,time_channels : int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels*2, out_channels,time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1))
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t: torch.Tensor):
        #print("x1 : ",x1.size())
        x1 = self.upconv(x1)
        #print("after upx1 : ",x1.size())
        x = torch.cat([x2, x1], dim=1)
        #print("catx : ",x.size())

        x = self.res(x,t)
        x = self.attn(x)
        return x

"""
class Middle(Module):
    def __init__(self, n_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels)

    def forward(self, x, t):
        x = self.res1(x,t)
        x = self.attn(x)
        x = self.res2(x,t)
        return x
"""

class Middle(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        #pdb.set_trace()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x,t)
        x = self.attn(x)
        x = self.res2(x,t)
        return x



############################################################################
class GTL(nn.Module):

    def __init__(self,netconf={}, GTL_IC=1,GTL_OC=64,n_chan=64,num_cbr=2,act='ReLU',pooling='maxpool',upmodule='bilinear',stage=4,use_bnorm=True,stage_option='slim',concat_type='concat',**kwargs):
        super(GTL,self).__init__()
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

        elif act == 'identity':
            self.act = nn.Sequential()
        self.stage_option = stage_option
        self.stage=stage
        self.lambdas= np.array([0,0,0,0,0]).astype(np.float32) # lambdas for modulation
        if stage_option == 'slim':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
            elif self.stage == 2:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 0:
                self.out=double_conv(n_chan,GTL_OC ,num_cbr,use_bnorm)

            elif self.stage == 3:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:

                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)

                time_channels = n_chan
                self.time_emb = TimeEmbedding(n_chan)

                #pdb.set_trace()

                self.mid = Middle(n_chan, time_channels)
                #self.sadown3 = SA_Down(n_chan, n_chan,True)
                #self.sadown4 = SA_Down(n_chan, n_chan,True)
                self.saup1 = SA_Up(n_chan, n_chan, time_channels, True)
                #self.saup2 = SA_Up(n_chan, n_chan, True)
                self.tsadown4 = TSA_Down(n_chan, n_chan,time_channels,True)
            elif self.stage == 5:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 6:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up6 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down6 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)

        elif stage_option == 'outconv_slim':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            self.out = nn.Sequential(nn.Conv2d(in_channels=n_chan, out_channels=GTL_OC, kernel_size=1, padding=0))
            
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 3:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 5:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 6:
                self.up1 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up2 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up5 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.up6 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm,concat_type=concat_type)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down5 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down6 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)

            elif self.stage == 0:
                self.dconv= nn.Sequential(
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm),
                double_conv(n_chan, n_chan,num_cbr,use_bnorm)
            )
        elif stage_option == 'fat':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up3 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 4:
                self.up1 = Up(n_chan*16+n_chan*8, n_chan*8, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*8+n_chan*4, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4+n_chan*2, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2+n_chan, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan, n_chan*2,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down2 = Down(n_chan*2, n_chan*4,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down3 = Down(n_chan*4, n_chan*8,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down4 = Down(n_chan*8, n_chan*16,num_cbr=num_cbr,use_bnorm=use_bnorm)

        elif stage_option == 'shuffle':
            self.input = double_conv(GTL_IC, n_chan,num_cbr,use_bnorm)
            if self.stage == 1:
                self.down1 = Down(n_chan, n_chan,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up4 = Up(n_chan*2, n_chan,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
            elif self.stage == 2:
                self.up4 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
            elif self.stage == 3:
                self.up3 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up1 = Up(n_chan*8, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                # self.up1 = Up(n_chan*16, n_chan*8,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down3 = Down(n_chan*16, n_chan*8, num_cbr=num_cbr*2,use_bnorm=use_bnorm,pooling =pooling)
                # self.down4 = Down(n_chan*32, n_chan*16, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)

            elif self.stage == 4:
                self.up4 = Up(n_chan*2, n_chan, upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up3 = Up(n_chan*4, n_chan*2,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up2 = Up(n_chan*8, n_chan*4,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.up1 = Up(n_chan*16, n_chan*8,  upmodule=upmodule,num_cbr=num_cbr,use_bnorm=use_bnorm)
                self.down1 = Down(n_chan*4,n_chan*2, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down2 = Down(n_chan*8, n_chan*4, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down3 = Down(n_chan*16, n_chan*8, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
                self.down4 = Down(n_chan*32, n_chan*16, num_cbr=num_cbr,use_bnorm=use_bnorm,pooling =pooling)
        
    def forward(self, x):
        x1 = self.input(x)

            
        if self.stage == 1:
            x = self.down1(x1)
            x = self.up4(x, x1)
        elif self.stage == 0:
            x = self.out(x)
        elif self.stage == 2:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
        elif self.stage == 3:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        elif self.stage == 4:
            t = self.time_emb(torch.tensor([0]).to(torch.device("cuda:0")))
            #pdb.set_trace()
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            #x4 = self.sadown3(x3,0)
            #x5 = self.down4(x4)
            #x5 = self.sadown4(x4,0)
            x5 = self.tsadown4(x4,t)

            # Modification
            if any(self.lambdas != 0):
                print('##################')
                print('Texture modulation')
                print(self.lambdas)
                print('###################')
                x1,x2,x3,x4,x5 =x1+self.lambdas[0] ,x2+self.lambdas[1],\
                    x3+self.lambdas[2],x4+self.lambdas[3],x5+self.lambdas[4]
            ##############################################
            #pdb.set_trace()
            x = self.saup1(x5,x4,t)
            #x = self.up1(x5, x4)
            #print("saup1 : ",x.size())
            #x = self.saup2(x, x3,0)
            x = self.up2(x, x3)
            #print("up2 : ",x.size())
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        elif self.stage == 5:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        elif self.stage == 6:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x = self.up1(x7, x6)
            x = self.up2(x, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)


        if self.stage_option =='outconv_slim':
            x = self.out(x)
        out = x
        return self.act(out)