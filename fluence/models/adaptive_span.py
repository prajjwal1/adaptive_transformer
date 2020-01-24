# The following code has been adapted from FAIR's "Adaptive Attention Span in Transformers " (ACL 2019) work with modifications

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# def _skew(X, pad_value):
#     """shift every row 1 step to right"""
#     # X = B x M x L
#     B, M, L = X.size()
#     X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
#     X = X.view(B, -1)  # B x ML+MM+M
#     X = X[:, :-M]  # B x ML+MM
#     X = X.view(B, M, M + L)  # B x M x L+M
#     return X


# def _unskew(X):
#     """reverse _skew operation"""
#     # X = B x M x L+M
#     B, M, L = X.size()
#     L -= M
#     X = X.view(B, -1)  # B x ML+MM
#     X = F.pad(X, (0, M))  # B x ML+MM+M
#     X = X.view(B, M, M + L + 1)  # B x M x L+M+1
#     X = X[:, :, :L]  # B x M x L
#     return X

def _unskew(X):
    """reverse _skew operation"""
    # X = B x H x (M x L+M)
    B, H, M, L = X.size()
    L -= M
    X = X.view(B*H, -1)  # (BxH) x (ML+MM)
    X = F.pad(X, (0, M))  # (BxH) x (ML+MM+M)
    X = X.view(B, H, M, M + L + 1)  # B X H x M x L+M+1
    X = X[:,:, :, :L]  # B x M x L
    return X

def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x H x (M x L)
    B, H, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B*H, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, H,M, M + L)  # B x M x L+M
    return X

class AdaptiveSpan(nn.Module):
    
    def __init__(self, attn_span, adapt_span_ramp, bs, nb_heads, mask_size, adapt_span_cache,
                adapt_span_loss_coeff, adapt_span_init, adapt_span_enabled):
        
        super(AdaptiveSpan,self).__init__()
        self.attn_span = attn_span    # [attn_span]
        self.ramp_size = adapt_span_ramp
        self.bs = bs
        self.nb_heads = nb_heads
        self.init_val = nn.Parameter(torch.Tensor([adapt_span_init]))
        self.adapt_cache = adapt_span_cache
        self.loss_coeff = adapt_span_loss_coeff
        self.shape = (self.bs, self.nb_heads,1, 1)
       
        self.current_val = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(*self.shape)) + self.init_val) # [bs,nb_heads,1,1]
        self.mask_size = mask_size
        
        mask_template_0 = torch.linspace(1 - self.mask_size[0], 0, steps=self.mask_size[0]) # [attn_span]
        mask_template_1 = torch.linspace(1 - self.mask_size[1], 0, steps=self.mask_size[1])
        self.register_buffer('mask_template_0', mask_template_0)
        self.register_buffer('mask_template_1', mask_template_1)

    def mask_forward(self,x):
        mask_size = x.size(3)
        if mask_size==self.mask_size[0]:
            mask = self.mask_template_0 + self.current_val*mask_size
        else:
            mask = self.mask_template_1 + self.current_val*mask_size
        mask = mask / self.ramp_size + 1                             
        mask = mask.clamp(0, 1)
        
        #if x.size(-1) < self.attn_span:
            # the input could have been trimmed beforehand to save computation
        #    mask = mask[:, :, -x.size(-1):]
        #assert x.size(3)==mask.size(3)
        #print(x.shape, mask.shape)
        x = x * mask   # [128, 12, 36, 64]) [128, 12, 1, 64]
        return x
    
    def get_current_avg_span(self,include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self.attn_span)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size

    def get_current_max_span(self,include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self.attn_span)
        if include_ramp:
            current_size += ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size
    
    
    def clamp_param(self):
        self.current_val.data.clamp_(0, 1)


    
    def get_trim_len(self):
        L = self.attn_span
        trim_len = min(L - 1, L - self.get_current_max_span())
        # too fine granularity might be bad for the memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len
    
    def trim_memory(self, query, key, value, key_pe):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self.attn_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            # cache is too short! this happens when validation resumes
            # after a lot of updates.
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe
    
    def get_cache_size(self):
        """determine how long the cache should be"""
        if self.adapt_cache:
            trim_len = self.get_trim_len()
            # give a buffer of 64 steps since a span might increase
            # in future updates
            return min(self.attn_span, self.attn_span - trim_len + 64)
        else:
            return self.attn_span
        
    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self.loss_coeff * self.attn_span * self.current_val.mean()
    
    def forward(self,attn):
        attn = self.mask_forward(attn)
        attn = attn/(attn.sum(-1,keepdim=True)+1e-8)
        return attn
    
# class AdaptiveMask(nn.Module):
#     """Soft masking function for adaptive size.
#     It masks out the last K values of an input. The masking value
#     goes from 1 to 0 gradually, so K can be learned with
#     back-propagation.
#     Args:
#         max_size: maximum size (i.e. input dimension)
#         ramp_size: size of the ramp going from 0 to 1
#         init_val: initial size proportion not to be masked out
#         shape: learn multiple sizes independent of each other
#     """

#     def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
#         nn.Module.__init__(self)
#         self._max_size = max_size    # [attn_span]
#         self._ramp_size = ramp_size
#         self.current_val = nn.Parameter(torch.zeros(*shape) + init_val) # [bs,nb_heads,1,1]
#         max_size = max_size
#         mask_template = torch.linspace(1 - max_size, 0, steps=max_size) # [attn_span]
#         self.register_buffer('mask_template', mask_template)

#     def forward(self, x):
#         mask_size = self._max_size
#         mask = self.mask_template + self.current_val * mask_size
#         mask = mask / self._ramp_size + 1                             
#         mask = mask.clamp(0, 1)
#         if x.size(-1) < self._max_size:
#             # the input could have been trimmed beforehand to save computation
#             mask = mask[:, :, -x.size(-1):]
#         assert x.size(3)==mask.size(3)
#         x = x * mask   # [128, 12, 36, 64]) [128, 12, 1, 64]
#         return x

#     def get_current_max_size(self, include_ramp=True):
#         current_size = math.ceil(self.current_val.max().item() * self._max_size)
#         if include_ramp:
#             current_size += self._ramp_size
#         current_size = max(0, min(self._max_size, current_size))
#         return current_size

#     def get_current_avg_size(self, include_ramp=True):
#         current_size = math.ceil(self.current_val.mean().item() * self._max_size)
#         if include_ramp:
#             current_size += self._ramp_size
#         current_size = max(0, min(self._max_size, current_size))
#         return current_size

#     def clamp_param(self):
#         """this need to be called after each update"""
#         self.current_val.data.clamp_(0, 1)


# class AdaptiveSpan(nn.Module):
#     """Adaptive attention span for Transformerself.
#     This module learns an attention span length from data for each
#     self-attention head.
#     Args:
#         attn_span: maximum attention span
#         adapt_span_loss: loss coefficient for the span length
#         adapt_span_ramp: length of the masking ramp
#         adapt_span_init: initial size ratio
#         adapt_span_cache: adapt cache size to reduce memory usage
#     """
#     def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
#                  adapt_span_init, adapt_span_cache, nb_heads, bs,**kargs):
#         nn.Module.__init__(self)
#         self._adapt_cache = adapt_span_cache
#         self._max_span = attn_span
#         self._loss_coeff = adapt_span_loss
#         self._nb_heads = nb_heads
#         self._mask = AdaptiveMask(max_size=self._max_span,
#                                  ramp_size=adapt_span_ramp,
#                                  init_val=adapt_span_init,
#                                  shape=(bs,nb_heads,1, 1)) 
        
#     def forward(self, attn):
#         """mask attention with the right span"""
#         # batch and head dimensions are merged together, so separate them first
#         B = attn.size(0) # batch size
#         M = attn.size(1) # block size
#         attn = self._mask(attn)
#         attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1

#         #attn = attn.view(B, M, -1)
#         return attn

#     def get_trim_len(self):
#         """how much of memory can be trimmed to reduce computation"""
#         L = self._max_span
#         trim_len = min(L - 1, L - self._mask.get_current_max_size())
#         # too fine granularity might be bad for the memory management
#         trim_len = math.floor(trim_len / 64) * 64
#         return trim_len

#     def trim_memory(self, query, key, value, key_pe):
#         """trim out unnecessary memory beforehand to reduce computation"""
#         trim_len = self.get_trim_len()
#         cache_size = key.size(1) - query.size(1)
#         trim_len_cache = trim_len - (self._max_span - cache_size)
#         if trim_len_cache > 0:
#             key = key[:, trim_len_cache:, :]
#             value = value[:, trim_len_cache:, :]
#         elif trim_len_cache < 0:
#             # cache is too short! this happens when validation resumes
#             # after a lot of updates.
#             key = F.pad(key, [0, 0, -trim_len_cache, 0])
#             value = F.pad(value, [0, 0, -trim_len_cache, 0])
#         if trim_len > 0:
#             if key_pe is not None:
#                 key_pe = key_pe[:, :, trim_len:]
#         return key, value, key_pe

#     def get_cache_size(self):
#         """determine how long the cache should be"""
#         if self._adapt_cache:
#             trim_len = self.get_trim_len()
#             # give a buffer of 64 steps since a span might increase
#             # in future updates
#             return min(self._max_span, self._max_span - trim_len + 64)
#         else:
#             return self._max_span

#     def get_loss(self):
#         """a loss term for regularizing the span length"""
#         return self._loss_coeff * self._max_span * self._mask.current_val.mean()

#     def get_current_max_span(self):
#         return self._mask.get_current_max_size()

#     def get_current_avg_span(self):
#         return self._mask.get_current_avg_size()

#     def clamp_param(self):
#         self._mask.clamp_param()