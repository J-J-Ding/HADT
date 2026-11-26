import math
import torch
from torch import nn
from torch.nn import functional as F
class LinearConv1d(nn.Conv1d):
    """Same zero-padded dilated cross-correlation, using GEMM on short patch sequences."""
    def forward(self, x):
        kernel = self.kernel_size[0]
        dilation = self.dilation[0]
        padded = F.pad(x, (self.padding[0], self.padding[0]))
        windows = padded.unfold(-1, 1 + (kernel - 1) * dilation, self.stride[0])[..., ::dilation]
        windows = windows.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[1] * kernel)
        return F.linear(windows, self.weight.reshape(self.out_channels, -1), self.bias).transpose(1, 2)


class ResidualTokenBlock(nn.Module):
    """Mix features with an MLP or neighboring patch positions with a CNN."""
    def __init__(self, width, hidden, dropout, convolution=False, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.convolution = convolution
        if convolution:
            self.net = nn.Sequential(
                LinearConv1d(width, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(dropout), LinearConv1d(hidden, width, kernel_size=1))
        else:
            self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        if self.convolution:
            z = self.net(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.net(z)
        return x + self.dropout(z)
