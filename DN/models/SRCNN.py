
import torch
import torch.nn as nn
from .operations import *
class SRCNN(nn.Module):
    """
    SRCNN with BatchNormalization
    """

    def __init__(self,aaa):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.out=Upsampler(default_conv, 2, n_feat=32 ,act=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.activate = nn.ReLU(inplace=True)
        rgb_range = 255
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
    def forward(self, x):
        x = self.sub_mean(x)
        h = self.activate(self.bn1(self.conv1(x)))
        h = self.activate(self.bn2(self.conv2(h)))
        h = self.out(h)
        out = self.conv3(h)
        out = self.add_mean(out)
        return out