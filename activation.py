import torch
from torch import nn
from torch.nn.functional import gelu


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)
