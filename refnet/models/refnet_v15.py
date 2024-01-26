import torch

from refnet.models.basemodel import CustomizedLDM, zero_drop
from refnet.models.cldm import ControlLDM
from refnet.modules.lora import RefLoraModules
from refnet.pooling import latent_shuffle
from sgm.util import exists, default, append_dims


class RefNet(CustomizedLDM):
    @torch.no_grad()
    def get_input(
            self,
            batch,
            k,
            bs=None,
            return_x=False,
            return_original_cond=False,
            shuffle_ref=True,
            *args,
            **kwargs
    ):
        if exists(bs):
           for key in batch.keys():
               batch[key] = batch[key][:bs]

        x = batch[self.first_stage_key]
        xc = batch[self.cond_stage_key]

        x, xc = map(
            lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
            (x, xc)
        )

        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
        c = self.get_learned_conditioning(xc).detach()
        if shuffle_ref:
            c = latent_shuffle(c)
        if self.training:
            c = zero_drop(c, self.ucg_rate) * c

        out = [z, dict(c_crossattn=[c])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc])
        return out

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
            shuffle_ref = False,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x, xc = out
            log["inputs"] = x
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
        else:
            z, c = out

        crossattn = c["c_crossattn"][0]
        B, _, H, W = z.shape
        uc_cross = self.get_unconditional_conditioning(crossattn)
        uc_full = {"c_crossattn": [uc_cross]}
        samples_cfg = self.sample(
            steps = steps,
            bs = B,
            shape = (self.channels, H, W),
            cond = c,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            unconditional_guidance_label = unconditional_guidance_label,
            device = z.device,
        )
        x_samples_cfg = self.decode_first_stage(samples_cfg.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log

    @torch.no_grad()
    def generate(
            self,
            sampler,
            step: int,
            gs: dict,
            bs: int,
            cond: dict,
            height: int = 512,
            width: int = 512,
            low_vram: bool = True,
    ):
        if low_vram:
            self.low_vram_shift("cond")

        emb = self.get_learned_conditioning(cond.get("crossattn"))
        null_emb = self.get_unconditional_conditioning(emb)

        if low_vram:
            self.low_vram_shift("unet")

        gs = gs.get("reference", 1)
        z = self.sample(
            cond = {"c_crossattn": [emb]},
            uncond = {"c_crossattn": [null_emb]},
            bs = bs,
            shape = (self.channels, height, width),
            cfg_scale = gs,
            step = step,
            sampler = sampler
        )

        if low_vram:
            self.low_vram_shift("first")
        return self.decode_first_stage(z)


class ControlledRefNet(RefLoraModules, ControlLDM):
    def __init__(self, r, **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        super().__init__(**kwargs)
        self.r = r
        self.is_xl = False
        self.controlled = True

        if exists(ckpt_path):
            print("Load base model before injecting lora modules.")
            self.init_from_ckpt(ckpt_path, logging=True)
        self.inject_lora_modules()

    def p_losses(self, x_start, cond, t, noise=None):
        # noisy training
        c = cond["c_crossattn"][0]
        c = self.q_sample(x_start=c, t=t, noise=torch.randn_like(c))
        cond["crossattn"] = [c]

        # loss calculation is refined according to sdxl
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        w = append_dims(self.sqrt_recipm1_alphas_cumprod[t] ** -2, x_start.ndim)
        model_output = self.predict_start_from_noise(
            x_t = x_noisy,
            t = t,
            noise = self.apply_model(x_noisy, t, cond)
        )

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        target = x_start

        loss = torch.mean(w * (model_output - target) ** 2)
        loss_dict.update({f'{prefix}/loss_simple': loss})
        return loss, loss_dict

    def get_input(self, batch, k, bs=None, return_x=False, return_original_cond=False,
                  *args, **kwargs):
        if exists(bs):
           for key in batch.keys():
               batch[key] = batch[key][:bs]

        x = batch[self.first_stage_key]
        xc = batch[self.cond_stage_key]
        control = batch[self.control_key]

        x, xc, control = map(
            lambda t: t.to(memory_format=torch.contiguous_format),
            (x, xc, control)
        )

        z = self.get_first_stage_encoding(self.encode_first_stage(x))
        c = self.get_learned_conditioning(xc)

        out = [z, dict(c_crossattn=[c], c_concat=[control])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc])
        return out

    def log_images(self, batch, N=4, n_row=2, sample=False, steps=20, ddim_eta=0.0,
                   unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, return_inputs=False, **kwargs):
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
            z, c, x, xc = out
            log["inputs"] = x
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
        else:
            z, c = out

        crossattn = c["c_crossattn"][0]
        B, _, H, W = z.shape
        if unconditional_guidance_scale > 1.0:
            uc_cross = torch.zeros_like(crossattn)
            uc_full = {"c_crossattn": [uc_cross]}
            samples_cfg = self.sample(
                cond=c,
                batch_size=B,
                shape=(self.channels, H, W),
                steps=steps,
                uncond=uc_full,
                cfg_scale=unconditional_guidance_scale,
                unconditional_guidance_label=unconditional_guidance_label,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        else:
            samples = self.sample(
                cond=c,
                batch_size=B,
                shape=(self.channels, H, W),
                steps=steps,
            )
            x_samples_cfg = self.decode_first_stage(samples)
            log["samples"] = x_samples_cfg
        return log

    def switch_to_fp16(self):
        self.control_model.half()
        self.model.diffusion_model.input_blocks.half()
        self.model.diffusion_model.middle_block.half()
        self.model.diffusion_model.output_blocks.half()
        self.model.diffusion_model.dtype = torch.float16

    def switch_to_fp32(self):
        self.control_model.float()
        self.model.diffusion_model.input_blocks.float()
        self.model.diffusion_model.middle_block.float()
        self.model.diffusion_model.output_blocks.float()
        self.model.diffusion_model.dtype = torch.float32