import torch
from torch import nn
from entmax import entmax_bisect

class AlphaChooser(torch.nn.Module):

    def __init__(self, head_count):
        super(AlphaChooser, self).__init__()
        self.pre_alpha = nn.Parameter(torch.randn(head_count))

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1.01, max=2)
    
class EntmaxAlpha(nn.Module):

    def __init__(self, head_count, dim=0):
        super(EntmaxAlpha, self).__init__()
        self.dim = dim
        self.alpha_chooser = nn.Parameter(AlphaChooser(head_count)())

    def forward(self, att_scores):
        batch_size, head_count, query_len, key_len = att_scores.size()

        self.alpha = self.alpha_chooser
        expanded_alpha = self.alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1,nb_heads,1,1]
        expanded_alpha = expanded_alpha.expand((batch_size, -1, query_len,1))# [bs, nb_heads, query_len,1]
        p_star = entmax_bisect(att_scores, expanded_alpha)

        return p_star