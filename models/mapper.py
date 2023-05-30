import torch
import torch.nn as nn

from inspect import isfunction
from ldm.modules.attention import CrossAttention
from models.wrapper import gap


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class NonLinearSelfAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, dim, out_dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_out = nn.Linear(dim, out_dim)

    def forward(self, x):
        attn = torch.einsum("b i c, b j c -> b i j", x, x) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b n i, b i c -> b n c", attn, x)
        return self.to_out(out)


class Mapper(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        self.reweight = nn.Linear(dim, dim)
        # self.cxt_attn = CrossAttention(dim, dim+1, heads=1, dim_head=dim)

        self.relu = nn.ReLU()
        self.dim = dim

    def projection_parameters(self):
        return list(self.proj_in.parameters()) + list(self.proj_out.parameters())

    def mapping_parameters(self):
        return list(self.reweight.parameters())

    def project_in(self, x):
        return self.relu(self.proj_in(x)) + 1.

    def project_out(self, x):
        return self.proj_out(x)

    def forward(self, v, t):
        """
            t: [batch_size, 2, dim+1], concatenated original text and target text with scale
        """
        x = self.relu(self.proj_in(v)) + 1.
        cls_token = x[:, 0].unsqueeze(1)

        t = self.relu(self.proj_in(t)) + 1.
        base, target = t.chunk(2, dim=1)
        x = self.reweight(x/cls_token) * (target - base)
        # project back into the original latent space
        out = self.proj_out(x)
        return v + out