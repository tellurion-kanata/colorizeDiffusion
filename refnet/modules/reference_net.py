import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Union
from functools import partial

from refnet.modules.unet import (
    timestep_embedding,
    conv_nd,
    TimestepEmbedSequential,
    exists,
    ResBlock,
    linear,
    Downsample,
    zero_module,
    SelfTransformerBlock,
    SpatialTransformer,
)

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def hack_inference_forward(model):
    model.forward = InferenceForward.__get__(model, model.__class__)


def hack_unet_forward(unet):
    unet.original_forward = unet._forward
    unet._forward = enhanced_forward.__get__(unet, unet.__class__)


def restore_unet_forward(unet):
    if hasattr(unet, "original_forward"):
        unet._forward = unet.original_forward.__get__(unet, unet.__class__)
        del unet.original_forward


def modulation(x, scale, shift):
    return x * (1 + scale) + shift


def enhanced_forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        hs_fg: torch.Tensor = None,
        hs_bg: torch.Tensor = None,
        mask: torch.Tensor = None,
        threshold: Union[float|torch.Tensor] = None,
        control: torch.Tensor = None,
        context: torch.Tensor = None,
        style_modulations: torch.Tensor = None,
        **additional_context
):
    h = x.to(self.dtype)
    emb = emb.to(self.dtype)
    hs = []

    control_iter = iter(control)
    for idx, module in enumerate(self.input_blocks):
        h = module(h, emb, context, mask, **additional_context)

        if idx in self.hint_encoder_index:
            h += next(control_iter)
        hs.append(h)

    h = self.middle_block(h, emb, context, mask, **additional_context)

    for idx, module in enumerate(self.output_blocks):
        h_skip = hs.pop()

        if exists(mask) and exists(threshold):
            # inject foreground/background features
            B, C, H, W = h_skip.shape
            cm = F.interpolate(mask, (H, W), mode="bicubic")
            h = torch.cat([h, torch.where(
                cm > threshold,
                self.map_modules[idx](h_skip, hs_fg[idx]) if exists(hs_fg) else h_skip,
                self.warp_modules[idx](h_skip, hs_bg[idx]) if exists(hs_bg) else h_skip
            )], 1)

        else:
            h = torch.cat([h, h_skip], 1)

        h = module(h, emb, context, mask, **additional_context)

        if exists(style_modulations):
            style_norm, emb_proj, style_proj = self.style_modules[idx]
            style_m = style_modulations[idx] + emb_proj(emb)
            style_m = style_proj(style_norm(style_m))[...,None,None]
            scale, shift = style_m.chunk(2, dim=1)

            h = modulation(h, scale, shift)

    return h

def enhanced_forward_xl(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        hs_fg: torch.Tensor = None,
        hs_bg: torch.Tensor = None,
        mask: torch.Tensor = None,
        inject_mask: torch.Tensor = None,
        threshold: Union[float|torch.Tensor] = None,
        control: torch.Tensor = None,
        context: torch.Tensor = None,
        style_modulations: torch.Tensor = None,
        **additional_context
):
    h = x.to(self.dtype)
    emb = emb.to(self.dtype)
    hs = []
    control_iter = iter(control)

    for idx, module in enumerate(self.input_blocks):
        h = module(h, emb, context, mask, **additional_context)

        if idx in self.hint_encoder_index:
            h += next(control_iter)
        hs.append(h)

    h = self.middle_block(h, emb, context, mask, **additional_context)

    for idx, module in enumerate(self.output_blocks):
        h_skip = hs.pop()

        if exists(inject_mask) and exists(threshold):
            # inject foreground/background features
            B, C, H, W = h_skip.shape
            cm = F.interpolate(inject_mask, (H, W), mode="bicubic")
            h = torch.cat([h, torch.where(
                cm > threshold,
                
                # foreground injection
                rearrange(
                    self.map_modules[idx][0](
                        rearrange(h_skip, "b c h w -> b (h w) c"),
                        hs_fg[idx] + self.map_modules[idx][1](emb).unsqueeze(1)
                    ), "b (h w) c -> b c h w", h=H, w=W
                ) + h_skip if exists(hs_fg) else h_skip,

                # background injection
                rearrange(
                    self.warp_modules[idx][0](
                        rearrange(h_skip, "b c h w -> b (h w) c"),
                        hs_bg[idx] + self.warp_modules[idx][1](emb).unsqueeze(1)
                    ), "b (h w) c -> b c h w", h=H, w=W
                ) + h_skip if exists(hs_bg) else h_skip
            )], 1)

        else:
            h = torch.cat([h, h_skip], 1)

        h = module(h, emb, context, mask, **additional_context)

        if exists(style_modulations):
            style_norm, emb_proj, style_proj = self.style_modules[idx]
            style_m = style_modulations[idx] + emb_proj(emb)
            style_m = style_proj(style_norm(style_m))[...,None,None]
            scale, shift = style_m.chunk(2, dim=1)

            h = modulation(h, scale, shift)

    return h

def InferenceForward(self, x, timesteps=None, y=None, *args, **kwargs):
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb).to(self.dtype)
    assert (y is not None) == (
            self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y.to(self.dtype))
    emb = emb.to(self.dtype)
    return self._forward(x, emb, *args, **kwargs)


class UNetEncoderXL(nn.Module):
    transformers = {
        "vanilla": SpatialTransformer,
    }

    def __init__(
            self,
            in_channels,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout = 0,
            channel_mult = (1, 2, 4, 8),
            conv_resample = True,
            dims = 2,
            num_classes = None,
            use_checkpoint = False,
            num_heads = -1,
            num_head_channels = -1,
            use_scale_shift_norm = False,
            resblock_updown = False,
            use_spatial_transformer = False,  # custom transformer support
            transformer_depth = 1,  # custom transformer support
            context_dim = None,  # custom transformer support
            disable_self_attentions = None,
            disable_cross_attentions = None,
            num_attention_blocks = None,
            use_linear_in_transformer = False,
            adm_in_channels = None,
            transformer_type = "vanilla",
            style_modulation = False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert exists(
                context_dim) or disable_cross_attentions, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
            assert transformer_type in self.transformers.keys(), f'Assigned transformer is not implemented.. Choices: {self.transformers.keys()}'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        time_embed_dim = model_channels * 4
        resblock = partial(
            ResBlock,
            emb_channels=time_embed_dim,
            dropout=dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        transformer = partial(
            self.transformers[transformer_type],
            context_dim=context_dim,
            use_linear=use_linear_in_transformer,
            use_checkpoint=use_checkpoint,
            disable_self_attn=disable_self_attentions,
            disable_cross_attn=disable_cross_attentions,
        )

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.style_modulation = style_modulation

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]

        time_embed_dim = model_channels * 4
        zero_conv = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_layers = nn.ModuleList([zero_module(
            nn.Linear(model_channels, model_channels * 2) if style_modulation else
            zero_conv(model_channels, model_channels)
        )])

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, num_head_channels)
                            if not use_spatial_transformer
                            else transformer(
                                ch, num_heads, num_head_channels, depth=transformer_depth[level]
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_layers.append(zero_module(
                    nn.Linear(ch, ch * 2) if style_modulation else zero_conv(ch, ch)
                ))

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    ))
                self.zero_layers.append(zero_module(
                    nn.Linear(out_ch, min(model_channels * 8, out_ch * 4)) if style_modulation else
                    zero_conv(out_ch, out_ch)
                ))
                ch = out_ch
                ds *= 2


    def forward(self, x, timesteps = None, y = None, *args, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y.to(self.dtype))

        hs = self._forward(x, emb, *args, **kwargs)
        return hs

    def _forward(self, x, emb, context = None, **additional_context):
        hints = []
        h = x.to(self.dtype)

        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context, **additional_context)

            if self.style_modulation:
                hint = self.zero_layers[idx](h.mean(dim=[2, 3]))
                hints.append(hint)

            else:
                hint = self.zero_layers[idx](h)
                hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
                hints.append(hint)

        hints.reverse()
        return hints