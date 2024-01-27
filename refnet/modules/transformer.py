from .attention import *
from torch.utils.checkpoint import checkpoint


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, disable_cross_attn=False, **kwargs):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.dim = dim
        self.disable_self_attn = disable_self_attn
        self.disable_cross_attn = disable_cross_attn

        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        if not disable_cross_attn:
            self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                                  heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        else:
            self.attn2 = None

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if not disable_cross_attn else None
        self.norm3 = nn.LayerNorm(dim)
        self.reference_scale = 1
        self.scale_factor = None

        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, emb=None, **kwargs):
        if self.checkpoint:
            return checkpoint(self._forward, x, context, mask, emb, use_reentrant=False)
        else:
            return self._forward(x, context, mask, emb)

    def _forward(self, x, context=None, mask=None, emb=None):
        x = self.attn1(self.norm1(x)) + x
        if not self.disable_cross_attn:
            x = self.attn2(self.norm2(x), context, mask, self.reference_scale, self.scale_factor) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SelfInjectedTransformerBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank = None
        self.time_proj = None
        self.injection_type = "concat"
        self.forward_without_bank = super()._forward

    def _forward(self, x, context=None, mask=None, emb=None):
        if exists(self.bank):
            bank = self.bank
            if bank.shape[0] != x.shape[0]:
                bank = bank.repeat(x.shape[0], 1, 1)
            if exists(self.time_proj) and exists(emb):
                bank = bank + self.time_proj(emb).unsqueeze(1)
            x_in = self.norm1(x)

            x = self.attn1(
                x = x_in,
                context = torch.cat([x_in] + [bank], 1) if self.injection_type == "concat" else x_in + bank,
                # mask=mask,
                # scale_factor = self.scale_factor
            ) + x

            x = self.attn2(
                x = self.norm2(x),
                context = context,
                mask = mask,
                scale = self.reference_scale,
                scale_factor = self.scale_factor
            ) + x

            x = self.ff(self.norm3(x)) + x
        else:
            x = self.forward_without_bank(x, context, mask, emb)
        return x


class SelfTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(
            self,
            dim,
            dim_head = 64,
            dropout = 0.,
            mlp_ratio = 4,
            checkpoint = True,
            reshape = True
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.attn = attn_cls(query_dim=dim, heads=dim//dim_head, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            zero_module(nn.Linear(dim * mlp_ratio, dim))
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.reshape = reshape
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        return self._forward(x, context)

    def _forward(self, x, context=None):
        b, c, h, w = x.shape
        if self.reshape:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        x = self.attn(self.norm1(x), context if exists(context) else None) + x
        x = self.ff(self.norm2(x)) + x

        if self.reshape:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x


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

    def forward(self, x, context=None, ip_context=None, mask=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, context, ip_context, mask, use_reentrant=False)
        else:
            return self._forward(x, context, ip_context, mask)

    def _forward(self, x, context, ip_context, mask):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context, ip_context, scale=self.reference_scale, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class Transformer(nn.Module):
    transformer_type = {
        "vanilla": BasicTransformerBlock,
        "self-injection": SelfInjectedTransformerBlock,
        "ip-adapter": IPTransformerBlock,
    }
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=True, type="vanilla", **kwargs):
        super().__init__()
        transformer_block = self.transformer_type[type]
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        if isinstance(context_dim, list):
            if depth != len(context_dim):
                context_dim = depth * [context_dim[0]]

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

    def forward(self, x, context=None, mask=None, emb=None, *args, **additional_context):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, mask=mask, emb=emb, *args, **additional_context)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def SpatialTransformer(*args, **kwargs):
    return Transformer(type="vanilla", *args, **kwargs)

def SelfInjectTransformer(*args, **kwargs):
    return Transformer(type="self-injection", *args, **kwargs)

def IPTransformer(*args, **kwargs):
    return Transformer(type="ip-adapter", *args, **kwargs)