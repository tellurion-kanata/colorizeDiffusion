import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum

import os
import math
from einops import rearrange, repeat
from inspect import isfunction
from typing import Optional, Any
from refnet.util import exists

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
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

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op) * scale

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


class IPAttention(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, *args, **kwargs):
        super().__init__(query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, *args, **kwargs)
        inner_dim = dim_head * heads
        self.ip_to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.ip_to_v = nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context=None, ip_context=None, mask=None, scale=1.):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        ip_k = self.ip_to_k(ip_context)
        ip_v = self.ip_to_v(ip_context)

        b, _, _ = q.shape
        q, k, v, ip_k, ip_v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v, ip_k, ip_v),
        )

        # actually compute the attention, what we cannot get enough of
        out = (xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op) +
               xformers.ops.memory_efficient_attention(q, ip_k, ip_v, attn_bias=None, op=self.attention_op) * scale)

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
        return list(self.ip_to_k.parameters()) + list(self.ip_to_v.parameters())