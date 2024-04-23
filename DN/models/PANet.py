import torch.nn as nn
import torch
import torch.nn as nn
from .operations import *
class PANET(nn.Module):
    def __init__(self, confs):
        super(PANET, self).__init__()
        conv=default_conv
        n_resblocks = confs['n_resblocks']
        n_feats = confs['n_feats']
        scale = confs['scale']
        rgb_range = confs['rgb_range']
        n_colors = confs['n_colors']
        res_scale = confs['res_scale']
        self.mode = confs['mode']
        # n_resblocks =80
        # n_resblocks =32
        # n_feats = 64
        kernel_size = 3 
        # scale = 10
        # rgb_range=1
        # n_colors=3
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        msa = PyramidAttention()
        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, nn.PReLU(), res_scale=res_scale
            ) for _ in range(n_resblocks//2)
        ]
        m_body.append(msa)
        for i in range(n_resblocks//2):
            m_body.append(ResBlock(conv,n_feats,kernel_size,nn.PReLU(),res_scale=res_scale))
      
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        #m_tail = [
        #    Upsampler(conv, scale, n_feats, act=False),
        #    conv(n_feats, args.n_colors, kernel_size)
        #]
        m_tail = [
            conv(n_feats, n_colors, kernel_size)
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        if not self.training:
            out= self.forward_chop(x)
            # self.weight_list=np.array([weights,Pre_act])
            return out 

        return self.model_forward(x)
    def model_forward(self,x):
        if self.mode=='SR':
            x = self.sub_mean(x)
        x = self.head(x)
        
        res = self.body(x)
        
        res += x

        x = self.tail(res)
        if self.mode=='SR':
            x = self.add_mean(x)

        return x 

    def forward_chop(self, x, shave=10, min_size=6800,gpuID=2):
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

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

