# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from . import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, NetConfig):
        super(RDN, self).__init__()
        r = 2
        G0 = 64
        kSize = 3
        self.chop = NetConfig['chop']
        self.chop_size= NetConfig['chop_size']

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }['B']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d( NetConfig['image_channels'], G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(r),
            nn.Conv2d(G0, NetConfig['image_channels'], kSize, padding=(kSize-1)//2, stride=1)
        ])
    def forward(self, x):

        if self.chop and not self.training:
            out= self.forward_chop(x,min_size=self.chop_size)
            # self.weight_list=np.array([weights,Pre_act])
            return out 
            

        return self.model_forward(x)
    def model_forward(self,x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return self.UPNet(x)

    def forward_chop(self, x, shave=16, min_size=64*64,gpuID=2):
        # print('chop')
        n_GPUs = min(1, 4)
        b, c, h, w = x.size()
        #############################################
        # adaptive shave
        # corresponding to scaling factor of the downscaling and upscaling modules in the network
        # # max shave size
        # shave_h = h % 2
        # shave_w = w % 2
        # # get half size of the hight and width
        # h_half, w_half = h // 2, w // 2
        # # mod
        # # h_size, w_size = h_half + shave_h, w_half + shave_w
        # h_size, w_size = h_half , w_half 
        shave_scale = shave
        # max shave size
        shave_size_max = shave*3
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        ###############################################
        #h_size, w_size = adaptive_shave(h, w)
        if w * h <= min_size:
            output = self.model_forward(x)
        elif w_size * h_size <= min_size:
            lr_list = [
                x[:, :, 0:h_size, 0:w_size],
                x[:, :, 0:h_size, (w - w_size):w],
                x[:, :, (h - h_size):h, 0:w_size],
                x[:, :, (h - h_size):h, (w - w_size):w]]
            sr_list = []
            weight_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model_forward(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
            output = torch.ones(b, c, h, w, requires_grad=False).cuda(gpuID)
            output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
            output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
            output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
            output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        else:
            lr_list = [
                x[:, :, 0:h_size, 0:w_size],
                x[:, :, 0:h_size, (w - w_size):w],
                x[:, :, (h - h_size):h, 0:w_size],
                x[:, :, (h - h_size):h, (w - w_size):w]]
            sr_list= [
                self.forward_chop(patch, shave=shave, min_size=min_size) for patch in lr_list
            ]
            # sr_list = [ l[0] for l in lists]
            output = torch.ones(b, c, h, w, requires_grad=False).cuda(gpuID)
            output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
            output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
            output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
            output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        # weight[:, :, 0:h_half, 0:w_half] = weight_list[0][:, :, 0:h_half, 0:w_half]
        # weight[:, :, 0:h_half, w_half:w] = weight_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        # weight[:, :, h_half:h, 0:w_half] = weight_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        # weight[:, :, h_half:h, w_half:w] = weight_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output