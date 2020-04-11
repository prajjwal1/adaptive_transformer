import torch
from torch import nn


class LayerDrop_Bert(nn.Module):
    def __init__(self, module_list, layers_to_drop=2):
        super(LayerDrop_Bert, self).__init__()
        self.module_list = module_list
        self.layers_to_drop = layers_to_drop
        self.length = len(module_list)

    def forward(self, feats, attention_mask):
        x = torch.randint(0, self.length, (self.layers_to_drop,))
        for index, layer in enumerate(self.module_list):
            if index not in x:
                feats = layer(feats, attention_mask)
        return feats


class LayerDrop_Cross(nn.Module):
    def __init__(self, module_list, layers_to_drop=2):
        super(LayerDrop_Cross, self).__init__()
        self.module_list = module_list
        self.layers_to_drop = layers_to_drop
        self.length = len(module_list)

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        x = torch.randint(0, self.length, (self.layers_to_drop,))
        for index, layer in enumerate(self.module_list):
            if index not in x:
                lang_feats, visn_feats = layer(
                    lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
                )  #
        return lang_feats, visn_feats
