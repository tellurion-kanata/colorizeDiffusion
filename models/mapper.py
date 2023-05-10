import torch
import torch.nn as nn
import xformers

from inspect import isfunction


def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads, with {dim_head} dim per head")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

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
        # out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class Mapper(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, dim, dim_head=64):
        super().__init__()
        heads = dim // dim_head
        self.pwm_mlp = nn.Sequential(*[
            nn.Linear(dim*2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim*2)
        ])

        self.cxt_attn = MemoryEfficientCrossAttention(dim, dim+1, heads=heads, dim_head=dim_head)
        self.dim = dim

    def forward(self, v, t):
        """
            t: [batch_size, 2, dim+1], concatenated original text and target text with scale
        """
        cls_token = v[:, 0].unsqueeze(1)
        pwm = self.pwm_mlp(torch.cat([cls_token.repeat(1, v.shape[1]-1, 1), v[:, 1:]], dim=2))
        pwm_w, pwm_b = torch.chunk(pwm, 2, dim=2)
        v = cls_token * pwm_w + pwm_b
        return v