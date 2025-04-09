import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum

import os
import math
from einops import rearrange, repeat
from typing import Optional, Any
from refnet.util import exists, default

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
    attn_op = xformers.ops.memory_efficient_attention
except:
    XFORMERS_IS_AVAILBLE = False
    attn_op = F.scaled_dot_product_attention

# CrossAttn precision handling
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")



def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)



class MemoryEfficientAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, log=False, **kwargs):
        super().__init__()
        if log:
            print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
                  f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        self.bg_scale = 1.
        self.fg_scale = 1.
        self.merge_scale = 0.
        self.mask_threshold = 0.05


    def forward(self, x, context=None, mask=None, scale=1., scale_factor=None):
        context = default(context, x)

        if exists(mask):
            out = self.masked_forward(x, context, mask, scale, scale_factor)
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            out = self.attn_forward(q, k, v, scale)

        return self.to_out(out)

    def attn_forward(self, q, k, v, scale=1.):
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        out = attn_op(q, k, v) * scale
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return out

    def masked_forward(self, x, context, mask, scale=1., scale_factor=None):
        """
        This function is used for mask-guided cross-attention, used for non-lora cases.
        Args:
            x: Query input
            context: Key and value input
            mask: Character mask
            scale: Attention scale
            sacle_factor: Current latent size factor
        """

        def qkv_forward(x, context):
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            return q, k, v

        assert exists(scale_factor), "Scale factor must be assigned before masked attention"

        mask = rearrange(
            F.interpolate(mask, scale_factor=scale_factor, mode="bicubic"),
            "b c h w -> b (h w) c"
        ).contiguous()

        c1, c2 = context.chunk(2, dim=1)

        # Background region cross-attention
        q2, k2, v2 = qkv_forward(x, c2)
        bg_out = self.attn_forward(q2, k2, v2, scale) * self.bg_scale

        # Character region cross-attention
        q1, k1, v1 = qkv_forward(x, c1)
        fg_out = self.attn_forward(q1, k1, v1, scale) * self.fg_scale

        fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
        return torch.where(mask < self.mask_threshold, bg_out, fg_out)


class MultiModalAttention(MemoryEfficientAttention):
    def __init__(self, query_dim, context_dim_2, heads=8, dim_head=64, *args, **kwargs):
        super().__init__(query_dim, heads=heads, dim_head=dim_head, *args, **kwargs)
        inner_dim = dim_head * heads
        self.to_k_2 = nn.Linear(context_dim_2, inner_dim, bias=False)
        self.to_v_2 = nn.Linear(context_dim_2, inner_dim, bias=False)

    def forward(self, x, context=None, mask=None, scale=1.):
        if not isinstance(scale, list) and not isinstance(scale, tuple):
            scale = (scale, scale)
        assert len(context.shape) == 4, "Multi-modal attention requires different context inputs to be (b, m, n c)"
        context, context2 = context.chunk(2, dim=1)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        k2 = self.to_k_2(context2)
        v2 = self.to_k_2(context2)

        b, _, _ = q.shape
        q, k, v, k2, v2 = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v, k2, v2)
        )

        out = attn_op(q, k, v) * scale[0] + attn_op(q, k2, v2) * scale[1]

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

    def get_trainable_params(self):
        return list(self.to_k_2.parameters()) + list(self.to_k_2.parameters())