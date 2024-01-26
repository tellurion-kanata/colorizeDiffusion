import torch.nn as nn
from ldm.modules.attention import MemoryEfficientCrossAttention, zero_module


class GlobalProjection(nn.Module):
    def __init__(self, c_dim, heads, head_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.dim_head = head_dim
        self.head = (heads[0], heads[0] * heads[1])

        self.proj1 = nn.Linear(c_dim, head_dim * heads[0])
        self.proj2 = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(head_dim, c_dim * heads[1])),
        )
        self.norm = nn.LayerNorm(c_dim)

    def forward(self, x):
        residual = x
        x = self.proj1(x).reshape(-1, self.head[0], self.dim_head).contiguous()
        x = self.proj2(x).reshape(-1, self.head[1], self.c_dim).contiguous() + residual
        return self.norm(x)


class LocalProjection(nn.Module):
    def __init__(self, dim, dim_head=64):
        super().__init__()
        attn_heads = dim // dim_head

        self.attn = MemoryEfficientCrossAttention(dim, dim, heads=attn_heads)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            zero_module(nn.Linear(dim, dim))
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.proj(self.norm1(x))
        return self.norm2(x)