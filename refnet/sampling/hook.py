import torch
import torch.nn as nn

from refnet.modules.transformer import BasicTransformerBlock, SelfInjectTransformer, SelfInjectedTransformerBlock

"""
    This implementation refers to Multi-ControlNet, thanks for the authors
    Paper: Adding Conditional Control to Text-to-Image Diffusion Models
    Link: https://github.com/Mikubill/sd-webui-controlnet
"""

def exists(v):
    return v is not None

def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class AutoMachine():
    Read = "read"
    Write = "write"


"""
    This class controls the attentions of reference unet and denoising unet
"""
class ReferenceAttentionControl:
    writer_modules = []
    reader_modules = []
    def __init__(self, time_embed_ch, only_decoder=True):
        self.time_embed_ch = time_embed_ch
        self.trainable_layers = nn.ModuleList()
        self.only_decoder = only_decoder
        self.hooked = False

    def insert_time_emb_proj(self, unet):
        for module in torch_dfs(unet.output_blocks if self.only_decoder else unet):
            if isinstance(module, BasicTransformerBlock):
                module.time_proj = nn.Linear(self.time_embed_ch, module.dim)
                self.trainable_layers.append(module.time_proj)

    def register(self, mode, unet):
        def transformer_forward_write(self, x, context=None, mask=None, emb=None):
            x_in = self.norm1(x)
            x = self.attn1(x_in) + x

            if not self.disable_cross_attn:
                x = self.attn2(self.norm2(x), context) + x
            x = self.ff(self.norm3(x)) + x

            self.bank = x_in
            return x

        def transformer_forward_read(self, x, context=None, mask=None, emb=None):
            if exists(self.bank):
                bank = self.bank
                if bank.shape[0] != x.shape[0]:
                    bank = bank.repeat(x.shape[0], 1, 1)
                bank = bank + self.time_proj(emb).unsqueeze(1)
                x_in = self.norm1(x)

                x = self.attn1(
                    x = x_in,
                    context = torch.cat([x_in] + [bank], 1),
                    # mask = mask,
                    scale_factor = self.scale_factor
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
                x = self.original_forward(x, context, mask, emb)
            return x

        assert mode in ["write", "read"]

        if mode == "read":
            self.hooked = True
        for module in torch_dfs(unet.output_blocks if self.only_decoder else unet):
            if isinstance(module, BasicTransformerBlock):
                if mode == "write":
                    module.original_forward = module._forward
                    module._forward = transformer_forward_write.__get__(module, BasicTransformerBlock)
                    self.writer_modules.append(module)
                else:
                    if not isinstance(module, SelfInjectedTransformerBlock):
                        module.original_forward = module._forward
                        module._forward = transformer_forward_read.__get__(module, BasicTransformerBlock)
                    self.reader_modules.append(module)

    def update(self):
        for idx in range(len(self.writer_modules)):
            self.reader_modules[idx].bank = self.writer_modules[idx].bank

    def restore(self):
        for idx in range(len(self.writer_modules)):
            self.writer_modules[idx]._forward = self.writer_modules[idx].original_forward
            self.reader_modules[idx]._forward = self.reader_modules[idx].original_forward
            self.reader_modules[idx].bank = None
        self.hooked = False

    def clean(self):
        for idx in range(len(self.reader_modules)):
            self.reader_modules[idx].bank = None
        self.hooked = False

    def reader_restore(self):
        for idx in range(len(self.reader_modules)):
            self.reader_modules[idx]._forward = self.reader_modules[idx].original_forward
            self.reader_modules[idx].bank = None
        self.hooked = False

    def get_trainable_params(self):
        return list(self.trainable_layers.parameters())


"""
    This class is for self-injection inside the denoising unet 
"""
class UnetHook:
    def __init__(self):
        super().__init__()
        self.attention_auto_machine = AutoMachine.Read

    def enhance_reference(
            self,
            model,
            ldm,
            bs,
            s,
            r,
            style_cfg=0.5,
            control_cfg=0,
            gr_indice=None,
            injection=False,
            start_step=0,
    ):
        def forward(self, x, t, control, context, **kwargs):
            if 1 - t[0] / (ldm.num_timesteps - 1) >= outer.start_step:
                # Write
                outer.attention_auto_machine = AutoMachine.Write

                rx = ldm.q_sample(outer.r.cpu(), torch.round(t.float()).long().cpu()).cuda().to(x.dtype)
                self.original_forward(rx, t, control=outer.s, context=context, **kwargs)

                # Read
                outer.attention_auto_machine = AutoMachine.Read
            return self.original_forward(x, t, control=control, context=context, **kwargs)

        def hacked_basic_transformer_inner_forward(self, x, context=None, mask=None, emb=None):
            x_norm1 = self.norm1(x)
            self_attn1 = None
            if self.disable_self_attn:
                # Do not use self-attention
                self_attn1 = self.attn1(x_norm1, context=context)

            else:
                # Use self-attention
                self_attention_context = x_norm1
                if outer.attention_auto_machine == AutoMachine.Write:
                    self.bank.append(self_attention_context.detach().clone())
                    self.style_cfgs.append(outer.current_style_fidelity)
                if outer.attention_auto_machine == AutoMachine.Read:
                    if len(self.bank) > 0:
                        style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                        self_attn1_uc = self.attn1(x_norm1,
                                                   context=torch.cat([self_attention_context] + self.bank, dim=1))
                        self_attn1_c = self_attn1_uc.clone()
                        if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                            self_attn1_c[outer.current_uc_indices] = self.attn1(
                                x_norm1[outer.current_uc_indices],
                                context=self_attention_context[outer.current_uc_indices])
                        self_attn1 = style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                    self.bank = []
                    self.style_cfgs = []
                if self_attn1 is None:
                    self_attn1 = self.attn1(x_norm1, context=self_attention_context)

            x = self_attn1.to(x.dtype) + x
            x = self.attn2(self.norm2(x), context, mask, self.reference_scale, self.scale_factor) + x
            x = self.ff(self.norm3(x)) + x
            return x

        self.s = [s.repeat(bs, 1, 1, 1) * control_cfg for s in ldm.control_encoder(s)]
        self.r = r
        self.injection = injection
        self.start_step = start_step
        self.current_uc_indices = gr_indice
        self.current_style_fidelity = style_cfg

        outer = self
        model = model.diffusion_model
        model.original_forward = model.forward
        # TODO: change the class name to target
        model.forward = forward.__get__(model, model.__class__)
        all_modules = torch_dfs(model)

        for module in all_modules:
            if isinstance(module, BasicTransformerBlock):
                module._original_forward = module._forward
                module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.style_cfgs = []


    def restore(self, model):
        model = model.diffusion_model
        if hasattr(model, "original_forward"):
            model.forward = model.original_forward
            del model.original_forward

        all_modules = torch_dfs(model)
        for module in all_modules:
            if isinstance(module, BasicTransformerBlock) and hasattr(module, "_original_forward"):
                module._forward = module._original_forward
                del module._original_forward, module.bank, module.style_cfgs
            if hasattr(module, "original_forward"):
                module.forward = module.original_forward
                del module.original_forward