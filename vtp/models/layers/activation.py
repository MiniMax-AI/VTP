import torch
from torch import nn


class QuickGELU(nn.Module):
    """Quick GELU activation function.

    NOTE: This is slower than nn.GELU or nn.SiLU and uses more GPU memory.
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
