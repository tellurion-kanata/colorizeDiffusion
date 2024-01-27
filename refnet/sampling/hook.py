import torch
import torch.nn as nn

from ldm.modules.attention import BasicTransformerBlock


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

class UnetHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_auto_machine = AutoMachine.Read
        self.gn_auto_machine = AutoMachine.Read
        self.gn_auto_machine_weight = 1.0

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
            adain=False,
            gn_weight=1.0,
            start_step=0,
    ):
        def forward(self, x, t, concat, context, **kwargs):
            original_control_scale = self.control_scale
            if 1 - t[0] / (ldm.num_timesteps - 1) >= outer.start_step:
                # Write
                if outer.injection:
                    outer.attention_auto_machine = AutoMachine.Write
                if outer.adain:
                    outer.gn_auto_machine = AutoMachine.Write

                self.control_scale = outer.current_control_scale
                rx = ldm.q_sample(outer.r.cpu(), torch.round(t.float()).long().cpu()).cuda()
                self.original_forward(x=rx, timesteps=t, concat=outer.s, context=context, **kwargs)

                # Read
                outer.attention_auto_machine = AutoMachine.Read
                outer.gn_auto_machine = AutoMachine.Read
            self.control_scale = original_control_scale
            return self.original_forward(x=x, timesteps=t, concat=concat, context=context, **kwargs)

        def hacked_basic_transformer_inner_forward(self, x, context=None):
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
            x = self.attn2(self.norm2(x), context=context, scale=self.reference_scale) + x
            x = self.ff(self.norm3(x)) + x
            return x

        def hacked_group_norm_forward(self, *args, **kwargs):
            eps = 1e-6
            x = self.original_forward(*args, **kwargs)
            y = None
            if outer.gn_auto_machine == AutoMachine.Write:
                if outer.gn_auto_machine_weight > self.gn_weight:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    self.mean_bank.append(mean)
                    self.var_bank.append(var)
                    self.style_cfgs.append(outer.current_style_fidelity)
            if outer.gn_auto_machine == AutoMachine.Read:
                if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                    style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                    mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    y_uc = (((x - mean) / std) * std_acc) + mean_acc
                    y_c = y_uc.clone()
                    if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                        y_c[outer.current_uc_indices] = x.to(y_c.dtype)[outer.current_uc_indices]
                    y = style_cfg * y_c + (1.0 - style_cfg) * y_uc
                self.mean_bank = []
                self.var_bank = []
                self.style_cfgs = []
            if y is None:
                y = x
            return y.to(x.dtype)

        if injection or adain:
            self.s = [s.repeat(bs, 1, 1, 1) for s in ldm.control_encoder(s)]
            self.r = r
            self.injection = injection
            self.adain = adain
            self.start_step = start_step
            self.current_uc_indices = gr_indice
            self.current_style_fidelity = style_cfg
            self.current_control_scale = control_cfg

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

            gn_modules = [model.middle_block]
            model.middle_block.gn_weight = 0

            input_block_indices = [4, 5, 7, 8, 10, 11]
            for w, i in enumerate(input_block_indices):
                module = model.input_blocks[i]
                module.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
                gn_modules.append(module)

            output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]
            for w, i in enumerate(output_block_indices):
                module = model.output_blocks[i]
                module.gn_weight = float(w) / float(len(output_block_indices))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, 'original_forward', None) is None:
                    module.original_forward = module.forward
                module.forward = hacked_group_norm_forward.__get__(module, torch.nn.Module)
                module.mean_bank = []
                module.var_bank = []
                module.style_cfgs = []
                module.gn_weight *= 2
            self.gn_auto_machine_weight = gn_weight

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