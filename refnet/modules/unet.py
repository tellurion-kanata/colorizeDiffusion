import torch
import torch.nn as nn

from refnet.modules.attention import (
    SelfTransformerBlock,
    Transformer,
    SpatialTransformer,
    FuseTransformer,
    DiTTransformer,
    IPTransformer
)
from ldm.modules.diffusionmodules.openaimodel import (
    timestep_embedding,
    conv_nd,
    TimestepBlock,
    exists,
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


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, injects=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, Transformer):
                inject = injects.pop(0) if exists(injects) else None
                x = layer(x, context, inject)
            else:
                x = layer(x)
        return x



class UNetModel(nn.Module):
    transformers = {
        "vanilla": SpatialTransformer,
        "fuse": FuseTransformer,
        "dit": DiTTransformer,
        "ipadapter": IPTransformer,
    }
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_type="vanilla",
        only_decoder=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
            assert transformer_type in self.transformers.keys(), f'Assigned transformer is not implemented.. Choices: {self.transformers.keys()}'
        transformer = self.transformers[transformer_type]

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.only_decoder = only_decoder
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
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

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
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
        self._feature_size = model_channels
        input_block_chans = [model_channels]
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
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, dim_head)
                            if not use_spatial_transformer or only_decoder
                            else transformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SelfTransformerBlock(ch, dim_head) if not use_spatial_transformer
            else transformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, num_heads, dim_head) if not use_spatial_transformer
                            else transformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

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

    def _forward(self, x, emb, concat=None, context=None, **kwargs):
        hs = []
        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return h


class DualCondUNet(UNetModel):
    def __init__(self, c_channels, model_channels, *args, **kwargs):
        super().__init__(model_channels=model_channels, *args, **kwargs)
        """
            Semantic condition input blocks, implementation from ControlNet.
            Paper: Adding Conditional Control to Text-to-Image Diffusion Models
            Authors: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala
            Code link: https://github.com/lllyasviel/ControlNet
        """
        self.semantic_input_blocks = TimestepEmbedSequential(
            conv_nd(2, c_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, model_channels, 3, padding=1))
        )

        self.control_scale = 1.

    def _forward(self, x, emb, concat=None, context=None, y=None, **kwargs):
        hs = []
        hint = self.semantic_input_blocks(concat, emb, context)
        h = x.to(self.dtype)
        for module in self.input_blocks:
            if exists(hint):
                h = module(h, emb, context) + hint * self.control_scale
                hint = None
            else:
                h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return h

"""
    Deepspeed converts all model weights to fp16 during training.
    Need to change all hidden outputs to fp16.
    Inference wrappers recover the original forward process according to Stable Diffusion implementation.
"""


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

class InferenceUNetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = InferenceForward.__get__(self, self.__class__)

class InferenceDualCondUNet(DualCondUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = InferenceForward.__get__(self, self.__class__)


class MultiDualCondUNet(UNetModel):
    def __init__(self, proj_length=None, encoder_type="mult", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.control_scale = 1.
        self.hint_encoder_index = [0, 3, 6, 9, 11]

        self.ebottom_idx = [6, 7]
        self.dbottom_idx = [2, 3, 4]
        self.proj_length = proj_length

        if encoder_type in ["full", "nfull",]:
            self.hint_decoder_index = [0, 3, 6, 9]
        else:
            self.hint_decoder_index = []

    def _forward(self, x, emb, concat=None, context=None, injects=None, **kwargs):
        h = x.to(self.dtype)

        hs = []
        concat_iter = iter(concat)
        for idx, module in enumerate(self.input_blocks):
            if exists(self.proj_length):
                if idx in self.ebottom_idx:
                    ccontext = context[:, :self.proj_length]
                else:
                    ccontext = context[:, self.proj_length:]
            else:
                ccontext = context
            h = module(h, emb, ccontext, injects)

            if idx in self.hint_encoder_index:
                h += next(concat_iter) * self.control_scale
            hs.append(h)

        h = self.middle_block(h, emb, context, injects)

        for idx, module in enumerate(self.output_blocks):
            if exists(self.proj_length):
                if idx in self.dbottom_idx:
                    ccontext = context[:, :self.proj_length]
                else:
                    ccontext = context[:, self.proj_length:]
            else:
                ccontext = context
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, ccontext, injects)

            if idx in self.hint_decoder_index:
                h += next(concat_iter) * self.control_scale
        return h


class InferenceMultiCondUNet(MultiDualCondUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = InferenceForward.__get__(self, self.__class__)