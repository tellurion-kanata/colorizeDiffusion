from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from refnet.util import exists
from refnet.modules.transformer import BasicTransformerBlock, SelfInjectedTransformerBlock


class Lora_target:
    none = 0
    foreground = 1
    background = 2


def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def masked_forward(self, x, context, mask, scale=1., scale_factor= None):
    """
    This function hacks cross-attention layers.
    Args:
        x: Query input
        context: Key and value input
        mask: Character mask
        scale: Attention scale
        sacle_factor: Current latent size factor

    """
    def qkv_forward(x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return q, k, v

    assert exists(scale_factor), "Scale factor must be assigned before masked attention"

    mask = rearrange(
        F.interpolate(mask, scale_factor=scale_factor, mode="bicubic"),
        "b c h w -> b (h w) c"
    ).contiguous()

    c1, c2 = context.chunk(2, dim=1)

    # Background region cross-attention
    switch_lora(self, Lora_target.background if self.use_bg_lora else Lora_target.none)
    q2, k2, v2 = qkv_forward(x, c2)
    bg_out = self.attn_forward(q2, k2, v2, scale) * self.bg_scale

    # Character region cross-attention
    switch_lora(self, Lora_target.foreground if self.use_fg_lora else Lora_target.none)
    q1, k1, v1 = qkv_forward(x, c1)
    fg_out = self.attn_forward(q1, k1, v1, scale) * self.fg_scale

    fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
    return torch.where(mask > self.mask_threshold, fg_out, bg_out)


def switch_lora(self, v):
    self.to_q.mode = v
    self.to_k.mode = v
    self.to_v.mode = v


def masked_transformer_forward(self, x, context, mask=None, emb=None, **kwargs):
    if exists(mask):
        mask = rearrange(
            F.interpolate(mask, scale_factor=self.scale_factor, mode="bicubic"),
            "b c h w -> b (h w) c"
        ).contiguous()
        c1, c2 = context.chunk(2, dim=1)

        # Background region cross-attention
        self.switch_lora(Lora_target.background if self.use_bg_lora else Lora_target.none)
        self.reference_scale = self.bg_scale
        bg_out = self.original_forward(x, c2, emb=emb, **kwargs)

        # Character region cross-attention
        self.switch_lora(Lora_target.foreground if self.use_fg_lora else Lora_target.none)
        self.reference_scale = self.fg_scale
        fg_out = self.forward_without_bank(x, c1, **kwargs)

        fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
        return torch.where(mask > self.mask_threshold, fg_out, bg_out)
    else:
        return self.original_forward(x, context, emb=emb, **kwargs)


def switch_transformer_lora(self, v):
    for lora in self.loras:
        lora.mode = v


class LoraInjectedLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias = False,
            r = 4,
            scale = 1.,
            dropout_p = 0.,
    ):
        super().__init__(in_features, out_features, bias)
        if isinstance(r, float):
            r = int(r * out_features)
        self.lora_down = nn.Linear(in_features, r, bias=bias)
        self.lora_up = nn.Linear(r, out_features, bias=bias)

        self.dropout = nn.Dropout(dropout_p)
        self.scale = scale
        self.mode = Lora_target.background

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def get_trainable_parameters(self, *args, **kwargs):
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())

    def forward(self, input):
        output = super().forward(input)

        if self.mode == Lora_target.background:
            output += self.lora_up(self.dropout(self.lora_down(input))) * self.scale
        return output


class SplitLoraInjectedLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            br = 0.5,
            fr = 0.5,
            fg_scale = 1.,
            bg_scale = 1.,
            dropout_p = 0.,
    ):
        super().__init__(in_features, out_features, bias)
        mfeatures = min(in_features, out_features)
        fr = int(fr * mfeatures) if isinstance(fr, float) else fr
        br = int(br * mfeatures) if isinstance(br, float) else br

        self.lora_down_fg = nn.Linear(in_features, fr, bias=bias)
        self.lora_up_fg = nn.Linear(fr, out_features, bias=bias)

        self.lora_down_bg = nn.Linear(in_features, br, bias=bias)
        self.lora_up_bg = nn.Linear(br, out_features, bias=bias)

        self.dropout = nn.Dropout(dropout_p)
        self.mode = Lora_target.none
        self.fg_scale = fg_scale
        self.bg_scale = bg_scale

        nn.init.normal_(self.lora_down_fg.weight, std=1 / fr)
        nn.init.zeros_(self.lora_up_fg.weight)

        nn.init.normal_(self.lora_down_bg.weight, std=1 / br)
        nn.init.zeros_(self.lora_up_bg.weight)

    def get_trainable_parameters(self, target, *args, **kwargs):
        if target == "foreground":
            return list(self.lora_up_fg.parameters()) + list(self.lora_down_fg.parameters())
        elif target == "background":
            return list(self.lora_up_bg.parameters()) + list(self.lora_down_bg.parameters())
        else:
            return (list(self.lora_up_fg.parameters()) +
                    list(self.lora_up_bg.parameters()) +
                    list(self.lora_down_fg.parameters()) +
                    list(self.lora_down_bg.parameters()))

    def forward(self, input):
        output = super().forward(input)

        if self.mode == Lora_target.none:
            return output

        elif self.mode == Lora_target.foreground:
            lora_down = self.lora_down_fg
            lora_up = self.lora_up_fg
            scale = self.fg_scale

        else:
            lora_down = self.lora_down_bg
            lora_up = self.lora_up_bg
            scale = self.bg_scale

        output += lora_up(self.dropout(lora_down(input))) * scale
        return output


class LoraModules:
    def __init__(self, *args, **kwargs):
        self.loras = []
        self.inject_lora(*args, **kwargs)

    def inject_cross_attn(self, block, br, fr, split):
        module = block.attn2
        named_modules = module.named_modules()
        for name, layer in named_modules:
            parent = module
            if isinstance(layer, nn.Linear):
                if name in ["to_q", "to_k", "to_v"]:
                    if split:
                        _tmp_layer = SplitLoraInjectedLinear(
                            in_features=layer.weight.shape[1],
                            out_features=layer.weight.shape[0],
                            bias=exists(layer.bias),
                            fr=fr,
                            br=br,
                        )
                    else:
                        # This case only trains the background lora
                        _tmp_layer = LoraInjectedLinear(
                            in_features=layer.weight.shape[1],
                            out_features=layer.weight.shape[0],
                            bias=exists(layer.bias),
                            r=br,
                        )

                    parent._modules[name] = _tmp_layer
                    self.loras.append(parent._modules[name])

        module.use_fg_lora = True
        module.use_bg_lora = True
        module.masked_forward = masked_forward.__get__(module, module.__class__)
        module.switch_lora = switch_lora.__get__(module, module.__class__)
        self.modules.append(module)

    def inject_transformer(self, block, br, fr, split):
        block.loras = []
        modules = torch_dfs(block)
        for module in modules:
            parent = module
            named_children = module.named_children()
            for name, layer in named_children:
                if isinstance(layer, nn.Linear) and name.find("time_proj") == -1:
                    _tmp_layer = SplitLoraInjectedLinear(
                            in_features=layer.weight.shape[1],
                            out_features=layer.weight.shape[0],
                            bias=exists(layer.bias),
                            fr=fr,
                            br=br,
                    ) if split else LoraInjectedLinear(
                        in_features=layer.weight.shape[1],
                        out_features=layer.weight.shape[0],
                        bias=exists(layer.bias),
                        r=br,
                    )
                    parent._modules[name] = _tmp_layer
                    self.loras.append(parent._modules[name])
                    block.loras.append(parent._modules[name])

        block.use_fg_lora = True
        block.use_bg_lora = True
        block.merge_scale = 0
        block.injection_type = "add"
        block.original_forward = block._forward
        block._forward = masked_transformer_forward.__get__(block, block.__class__)
        block.switch_lora = switch_transformer_lora.__get__(block, block.__class__)
        self.modules.append(block)


    def inject_lora(self, unet, br, fr=4, split=True, only_crossattn=True):
        self.loras = []
        self.modules = []

        all_modules = torch_dfs(unet)
        for block in all_modules:
            if isinstance(block, SelfInjectedTransformerBlock) and not only_crossattn:
                self.inject_transformer(block, br, fr, split)
            elif isinstance(block, BasicTransformerBlock):
                self.inject_cross_attn(block, br, fr, split)

    def get_trainable_lora_weights(self, target="both"):
        assert target in ["foreground", "background", "both"]
        params = []
        for layer in self.loras:
            params += layer.get_trainable_parameters(target)
        return params

    def switch_lora(self, activate_fg, activate_bg):
        for module in self.modules:
            module.use_fg_lora = activate_fg
            module.use_bg_lora = activate_bg

    def adjust_lora_scales(self, fg_scale, bg_scale):
        for lora in self.loras:
            lora.fg_scale = fg_scale
            lora.bg_scale = bg_scale

    def deactivate_lora_weights(self):
        for lora in self.loras:
            lora.mode = Lora_target.none
            lora.mode = Lora_target.none