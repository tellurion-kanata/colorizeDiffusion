import torch
import torch.nn as nn

from ldm.modules.attention import BasicTransformerBlock
from einops import rearrange


def exists(val):
    return val is not None

class CondTransformer(nn.Module):
    def __init__(self, input_dim, context_dim, n_layers=4, inner_dim=None, head_dim=64, checkpoint=True):
        super().__init__()
        n_heads = inner_dim // head_dim

        self.proj_in = nn.Linear(input_dim, inner_dim)
        self.transformer = nn.ModuleList()
        for i in range(n_layers):
            self.model.append(
                BasicTransformerBlock(dim=input_dim, n_heads=n_heads, d_head=head_dim, checkpoint=checkpoint)
            )
        self.proj_out = nn.Linear(inner_dim, context_dim)

    def forward(self, x):
        return self.proj_out(self.model(self.proj_in(x)))


class PromptTransformer(nn.Module):
    def __init__(self, input_dim, context_dim, n_layers=4, inner_dim=None, head_dim=64, checkpoint=True):
        super().__init__()
        inner_dim = inner_dim if exists(inner_dim) else input_dim
        n_heads = inner_dim // head_dim
        self.embedding = nn.Linear(context_dim + 1, context_dim)

        self.proj_in = nn.Linear(input_dim, inner_dim)
        self.transformer = nn.ModuleList()
        for i in range(n_layers):
            self.model.append(
                BasicTransformerBlock(
                    dim=inner_dim, context_dim=context_dim, n_heads=n_heads,
                    d_head=head_dim, checkpoint=checkpoint
                )
            )
        self.proj_out = nn.Linear(inner_dim, input_dim)

    def forward(self, x, t, s):
        b, c, h, w = x.shape
        t_in = torch.cat([t, s], dim=1).unsqueeze(1)

        base = x
        context = self.embedding(t_in)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.proj_in(x)
        x = self.transformer(x, context)
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return base + x