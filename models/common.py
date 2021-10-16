import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class cha_loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(cha_loss, self).__init__()
        self.eps=eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target) ** 2 + self.eps ** 2
        out = torch.sqrt(diff)
        loss = torch.mean(out)

        return loss

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, :, index]

    return it

class UPSCALE(nn.Module): 
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.2,True)):
        super(UPSCALE, self).__init__()
        body = []
        body.append(nn.Conv2d(basic_feature, 48, 3, 1, 3//2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(12, 12, 3, 1, 3//2))
        body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class PFRB(nn.Module):
    '''
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    '''
    def __init__(self, basic_feature=64, num_channel=3, act=torch.nn.LeakyReLU(0.2,True)):
        super(PFRB, self).__init__()
        self.bf = basic_feature
        self.nc = num_channel
        self.act = act
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.bf, self.bf, 3, 1, 3//2) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.bf * num_channel, self.bf, 1, 1, 1//2)
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3//2) for _ in range(num_channel)])
    
    def forward(self, x):
        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nc)]
        merge = torch.cat(x1,1)
        base = self.act(self.conv1(merge))
        x2 = [torch.cat([base, i],1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i], x2[i]) for i in range(self.nc)]