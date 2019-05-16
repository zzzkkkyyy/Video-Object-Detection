import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, n_channels, gamma=None):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = gamma
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(x).sum(dim=1, keepdim=True).sqrt() + self.eps
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class EmbeddingNetwork(nn.Module):

    def __init__(self, in_channels):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)

        return x