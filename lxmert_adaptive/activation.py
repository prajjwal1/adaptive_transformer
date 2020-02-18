import torch
from torch.nn.functional import gelu
from torch import nn

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.nn.functional.gelu(x)