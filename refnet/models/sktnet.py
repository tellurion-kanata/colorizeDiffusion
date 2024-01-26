import torch

from sgm.util import exists
from refnet.models.basemodel import CustomizedLDM, zero_drop


class SktNet(CustomizedLDM):
    @torch.no_grad()
    def get_input(
            self,
            batch,
            k,
            bs=None,
            return_x=False,
            return_original_cond=False,
            *args,
            **kwargs
    ):
        if exists(bs):
           for key in batch.keys():
               batch[key] = batch[key][:bs]

        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        x, c = map(
            lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
            (x, c)
        )

        if self.training:
            # use blank image as unconditional input
            c = zero_drop(c, self.ucg_rate) * c
        c = (c - 0.5) * 2
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        out = [z, c]
        if return_x:
            out.extend([x])
        return out

    def get_unconditional_conditioning(self, x, null_label=None):
        return self.get_learned_conditioning(-torch.ones_like(x, device=x.device)).detach()

    @torch.no_grad()
    def log_images(
            self,
            batch,
            N=4,
            n_row=2,
            sample=False,
            steps=20,
            ddim_eta=0.0,
            unconditional_guidance_scale=9.0,
            unconditional_guidance_label=None,
            use_ema_scope=True,
            return_inputs=False,
            **kwargs
    ):
        out = self.get_input(
            batch,
            self.first_stage_key,
            bs = N,
            return_x = return_inputs,
            return_original_cond = return_inputs,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x = out
            log["inputs"] = x
            log["conditioning"] = c
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
        else:
            z, c = out

        B, _, H, W = z.shape
        concat = c
        c = dict(c_concat=[self.get_learned_conditioning(c)])
        uc_concat = self.get_unconditional_conditioning(concat)
        uc_full = {"c_concat": [uc_concat]}
        samples_cfg = self.sample(
            cond = c,
            batch_size = B,
            shape = (self.channels, H, W),
            steps = steps,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            unconditional_guidance_label = unconditional_guidance_label,
        )
        x_samples_cfg = self.decode_first_stage(samples_cfg.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log

    def switch_to_fp16(self):
        unet = self.model.diffusion_model

        self.cond_stage_model = self.cond_stage_model.half()
        unet.input_blocks = unet.input_blocks.half()
        unet.middle_block = unet.middle_block.half()
        unet.output_blocks = unet.output_blocks.half()
        unet.dtype = torch.float16

    def switch_to_fp32(self):
        unet = self.model.diffusion_model

        self.cond_stage_model = self.cond_stage_model.float()
        unet.input_blocks = unet.input_blocks.float()
        unet.middle_block = unet.middle_block.float()
        unet.output_blocks = unet.output_blocks.float()
        unet.dtype = torch.float32