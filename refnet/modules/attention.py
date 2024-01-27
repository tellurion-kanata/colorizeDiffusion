import torch
import torch.nn as nn
import xformers

from einops import rearrange
from ldm.util import exists
from ldm.modules.attention import (
    zero_module,
    checkpoint,
    Normalize,
    FeedForward,
    CrossAttention,
    MemoryEfficientCrossAttention,
    BasicTransformerBlock,
    XFORMERS_IS_AVAILBLE,
)


class SelfTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, dim_head=64, dropout=0., mlp_ratio=4, checkpoint=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.attn = attn_cls(query_dim=dim, heads=dim//dim_head, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x


"""
    Diffusion Transformer block.
    Paper: Scalable Diffusion Models with Transformers
    Arxiv: https://arxiv.org/abs/2212.09748
"""
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, dim_head, context_dim, mlp_ratio=4.0, dropout=0., checkpoint=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = attn_cls(dim, heads=n_heads, dim_head=dim_head)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, 6 * dim, bias=True)
        )

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(context).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FuseBlock(BasicTransformerBlock):
    def __init__(self, dim, context_dim=None, embed_dim=1280, *args, **kwargs):
        super().__init__(dim=dim, context_dim=context_dim, *args, **kwargs)
        attn_head = context_dim // 64
        self.latents = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.adaln_attn = MemoryEfficientCrossAttention(context_dim, context_dim=embed_dim, heads=attn_head)
        self.adaln_ff = zero_module(nn.Linear(embed_dim, 3 * context_dim))
        self.adaln_ca = zero_module(nn.Linear(embed_dim, 3 * dim))

        self.adaln_norm1 = nn.LayerNorm(embed_dim)
        self.adaln_norm2 = nn.LayerNorm(embed_dim)

    def _forward(self, x, context=None):
        b, _, c = context.shape
        local, cls = context[:, :256], context[:, 256:]
        latents = self.latents.repeat(b, 1, 1)
        latents = latents + self.adaln_attn(latents, local)
        shift_mca, scale_mca, gate_mca = self.adaln_ca(self.adaln_norm1(latents[:, :1])).chunk(3, dim=2)
        shift_mlp, scale_mlp, gate_mlp = self.adaln_ff(self.adaln_norm2(latents[:, 1:])).chunk(3, dim=2)

        x = x + self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)
        x = x + gate_mca * self.attn2(self.norm2(x), context=modulate(cls, shift_mca, scale_mca), scale=self.reference_scale)
        x = x + gate_mlp * self.ff(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


# IP-Adapter modules
class IPAttention(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim, ip_token_num=16, heads=8, dim_head=64, *args, **kwargs):
        super().__init__(query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, *args, **kwargs)
        inner_dim = dim_head * heads
        self.ip_token_num = ip_token_num
        self.ip_to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.ip_to_v = nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context=None, mask=None, scale=1.):
        ip, context = context[:, :self.ip_token_num], context[:, self.ip_token_num:]

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        ip_k = self.ip_to_k(ip)
        ip_v = self.ip_to_v(ip)

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


class IPTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                     disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = MemoryEfficientCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = IPAttention(query_dim=dim, context_dim=context_dim,
                                 heads=n_heads, dim_head=d_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.reference_scale = 1
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context, scale=self.reference_scale) + x
        x = self.ff(self.norm3(x)) + x
        return x


class Transformer(nn.Module):
    transformer_type = {
        "vanilla": BasicTransformerBlock,
        "dit": DiTBlock,
        "fuse": FuseBlock,
        "ip-adapter": IPTransformerBlock,
    }
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=True, type="vanilla", **kwargs):
        super().__init__()
        transformer_block = self.transformer_type[type]
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [transformer_block(inner_dim, n_heads, d_head,
                               dropout=dropout, context_dim=context_dim[d], checkpoint=use_checkpoint, **kwargs)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, inject=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], inject=inject)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def SpatialTransformer(*args, **kwargs):
    return Transformer(type="vanilla", *args, **kwargs)

def DiTTransformer(*args, **kwargs):
    return Transformer(type="dit", *args, **kwargs)

def IPTransformer(*args, **kwargs):
    return Transformer(type="ip-adapter", *args, **kwargs)

def FuseTransformer(*args, **kwargs):
    return Transformer(type="fuse", *args, **kwargs)