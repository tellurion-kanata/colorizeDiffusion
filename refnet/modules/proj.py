import torch
import torch.nn as nn

from refnet.util import default
from refnet.modules.attention import MemoryEfficientAttention, zero_module, FeedForward, exists
from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    def __init__(self, dim, output_dim, mlp_ratio=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, output_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(output_dim * mlp_ratio, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.model(x))


class NormalizedLinear(nn.Module):
    def __init__(self, dim, output_dim, checkpoint=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.checkpoint = checkpoint

    def forward(self, x):
        if self.checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, dim, output_dim=None, dim_head=64, dropout=0., checkpoint=True):
        super().__init__()
        output_dim = default(output_dim, dim)
        self.attn = MemoryEfficientAttention(dim, heads=dim//dim_head, dropout=dropout)
        self.ff = FeedForward(dim, dim_out=output_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.checkpoint = checkpoint

    def forward(self, x, emb=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        x = self.attn(x, context=emb) + x
        x = self.ff(self.norm1(x))
        return self.norm2(x)


class IPAdapterPlus(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.LayerNorm(dim),
                        nn.LayerNorm(dim),
                        MemoryEfficientAttention(dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.shape[0], 1, 1)

        x = self.proj_in(x)

        for norm1, norm2, norm3, attn, ff in self.layers:
            latents = attn(norm1(x), norm2(latents)) + latents
            latents = ff(norm3(latents)) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)



class GlobalProjection(nn.Module):
    def __init__(self, input_dim, output_dim, heads, dim_head=128, checkpoint=True):
        super().__init__()
        self.c_dim = output_dim
        self.dim_head = dim_head
        self.head = (heads[0], heads[0] * heads[1])

        self.proj1 = nn.Linear(input_dim, dim_head * heads[0])
        self.proj2 = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(dim_head, output_dim * heads[1])),
        )
        self.norm = nn.LayerNorm(output_dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        if self.checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = self.proj1(x).reshape(-1, self.head[0], self.dim_head).contiguous()
        x = self.proj2(x).reshape(-1, self.head[1], self.c_dim).contiguous()
        return self.norm(x)


class ClusterTransformer(nn.Module):
    def __init__(
            self,
            dim,
            token_length,
            dim_head = 64,
            dropout = 0.,
            mlp_ratio = 4,
            layers = 4,
            context_dim = None,
            projection_head = 0,
            checkpoint = True,
            skip = False,
    ):
        super().__init__()
        context_dim = context_dim or dim
        self.layers = nn.ModuleList()

        for i in range(layers):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    nn.LayerNorm(dim),
                    MemoryEfficientAttention(
                        query_dim = dim,
                        heads = dim//dim_head,
                        context_dim = context_dim,
                        dropout = dropout
                    ),
                    nn.Sequential(
                        nn.Linear(dim, dim * mlp_ratio),
                        nn.SiLU(),
                        nn.Linear(dim * mlp_ratio, dim)
                    )
                ])
            )
        self.embedding = nn.Parameter(torch.randn([token_length, dim]))
        self.skip = skip
        self.checkpoint = checkpoint

        if projection_head > 0:
            self.proj = nn.Linear(context_dim, projection_head * context_dim)
            self.pnorm = nn.LayerNorm(context_dim)
        else:
            self.proj = None

    def forward(self, x):
        if self.checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, emb):
        b, n, c = emb.shape

        if exists(self.proj):
            emb = self.proj(emb).reshape(b, -1, c).contiguous()
            emb = self.pnorm(emb)

        x = self.embedding.repeat(b, 1, 1)
        for norm1, norm2, attn, ff in self.layers:
            x = attn(norm1(x), emb) + x
            x = ff(norm2(x)) + x

        if self.skip:
            x = x + emb
        return x