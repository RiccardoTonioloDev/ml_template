from ml_template.models.components import ConvBlock
from torch import Tensor
from typing import cast
from torch import nn


class DummyModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.m = nn.Sequential(ConvBlock(3, 9), nn.Conv2d(9, 3, 3), nn.Flatten(1))

    def forward(self, x: Tensor):
        return cast(Tensor, self.m(x)).mean(1).unsqueeze(1)
