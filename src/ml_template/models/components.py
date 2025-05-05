from torch import Tensor
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor):
        return self.m(x)
