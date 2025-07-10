import torch
import torch.nn as nn

from functools import partial
from refnet.util import exists
from refnet.modules.transformer import (
    SelfTransformerBlock,
    Transformer,
    SpatialTransformer,
    rearrange
)
from refnet.ldm.openaimodel import (
    timestep_embedding,
    conv_nd,
    TimestepBlock,
    zero_module,
    ResBlock,
    linear,
    Downsample,
    Upsample,
    normalization,
)

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def hack_inference_forward(model):
    model.forward = InferenceForward.__get__(model, model.__class__)

def InferenceForward(self, x, timesteps=None, y=None, *args, **kwargs):
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb).to(self.dtype)
    assert (y is not None) == (
            self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y.to(emb.device))
    emb = emb.to(self.dtype)
    h = self._forward(x, emb, *args, **kwargs)
    return self.out(h.to(x.dtype))



class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, mask=None, **additional_context):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, Transformer):
                x = layer(x, context, mask, **additional_context)
            else:
                x = layer(x)
        return x



class UNetModel(nn.Module):
    transformers = {
        "vanilla": SpatialTransformer,
    }
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
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
            map_module = False,
            warp_module = False,
            style_modulation = False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert exists(context_dim) or disable_cross_attentions, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
            assert transformer_type in self.transformers.keys(), f'Assigned transformer is not implemented.. Choices: {self.transformers.keys()}'
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        assert num_heads > -1 or num_head_channels > -1, 'Either num_heads or num_head_channels has to be set'
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
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.num_classes = num_classes
        self.model_channels = model_channels
        self.dtype = torch.float32

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]
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

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [resblock(ch, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels > -1:
                        current_num_heads = ch // num_head_channels
                        current_head_dim = num_head_channels
                    else:
                        current_num_heads = num_heads
                        current_head_dim = ch // num_heads

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, current_head_dim)
                            if not use_spatial_transformer
                            else transformer(
                                ch, current_num_heads, current_head_dim, depth=transformer_depth[level]
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    resblock(ch, out_channels=out_ch, down=True) if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels > -1:
            current_num_heads = ch // num_head_channels
            current_head_dim = num_head_channels
        else:
            current_num_heads = num_heads
            current_head_dim = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            resblock(ch),
            SelfTransformerBlock(ch, current_head_dim) if not use_spatial_transformer
            else transformer(ch, current_num_heads, current_head_dim, depth=transformer_depth_middle),
            resblock(ch),
        )

        self.output_blocks = nn.ModuleList([])
        self.map_modules = nn.ModuleList([])
        self.warp_modules = nn.ModuleList([])
        self.style_modules = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [resblock(ch + ich, out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels > -1:
                        current_num_heads = ch // num_head_channels
                        current_head_dim = num_head_channels
                    else:
                        current_num_heads = num_heads
                        current_head_dim = ch // num_heads

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, current_head_dim) if not use_spatial_transformer
                            else transformer(
                                ch, current_num_heads, current_head_dim, depth=transformer_depth[level]
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        resblock(ch, up=True) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

                if map_module:
                    self.map_modules.append(
                        SelfTransformerBlock(ich)
                    )

                if warp_module:
                    self.warp_modules.append(
                        SelfTransformerBlock(ich)
                    )

                if style_modulation:
                    self.style_modules.append(nn.ModuleList([
                        nn.LayerNorm(ch*2),
                        nn.Linear(time_embed_dim, ch*2),
                        zero_module(nn.Linear(ch*2, ch*2))
                    ]))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, y=None, *args, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y.to(self.dtype))

        h = self._forward(x, emb, *args, **kwargs)
        return self.out(h).to(x.dtype)

    def _forward(self, x, emb, control=None, context=None, mask=None, **additional_context):
        hs = []
        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, mask, **additional_context)
            hs.append(h)

        h = self.middle_block(h, emb, context, mask, **additional_context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, mask, **additional_context)
        return h


class DualCondUNet(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_encoder_index = [0, 3, 6, 9, 11]

    def _forward(self, x, emb, control=None, context=None, mask=None, **additional_context):
        h = x.to(self.dtype)
        hs = []

        control_iter = iter(control)
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context, mask, **additional_context)

            if idx in self.hint_encoder_index:
                h += next(control_iter)
            hs.append(h)

        h = self.middle_block(h, emb, context, mask, **additional_context)

        for idx, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, mask, **additional_context)

        return h

class OldUnet(UNetModel):
    def __init__(self, c_channels, model_channels, channel_mult, *args, **kwargs):
        super().__init__(channel_mult=channel_mult, model_channels=model_channels, *args, **kwargs)
        """
            Semantic condition input blocks, implementation from ControlNet.
            Paper: Adding Conditional Control to Text-to-Image Diffusion Models
            Authors: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala
            Code link: https://github.com/lllyasviel/ControlNet
        """
        from refnet.modules.encoder import SimpleEncoder, MultiEncoder
        # self.semantic_input_blocks = SimpleEncoder(c_channels, model_channels)
        self.semantic_input_blocks = MultiEncoder(c_channels, model_channels, channel_mult)
        self.hint_encoder_index = [0, 3, 6, 9, 11]

    def forward(self, x, timesteps=None, control=None, context=None, y=None, **kwargs):
        concat = control[0].to(self.dtype)
        context = context.to(self.dtype)

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb).to(self.dtype)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.to(self.dtype)
        hints = self.semantic_input_blocks(concat, emb, context)

        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if idx in self.hint_encoder_index:
                h += hints.pop(0)

            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.to(x.dtype)
        return self.out(h)


class UNetEncoder(nn.Module):
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
                layers = [resblock(ch, out_channels=mult * model_channels)]
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
                    resblock(ch, out_channels=mult * model_channels, down=True) if resblock_updown else Downsample(
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

        for zero_layer, module in zip(self.zero_layers, self.input_blocks):
            h = module(h, emb, context, **additional_context)

            if self.style_modulation:
                hint = zero_layer(h.mean(dim=[2, 3]))
            else:
                hint = zero_layer(h)
                hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
            hints.append(hint)

        hints.reverse()
        return hints