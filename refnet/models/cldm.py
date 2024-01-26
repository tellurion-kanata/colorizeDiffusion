import torch

from sgm.util import instantiate_from_config, exists, log_txt_as_img
from refnet.models.basemodel import CustomizedColorizer, zero_drop


"""
    Forked from ControlNet v1.1 nightly
    Github: https://github.com/lllyasviel/ControlNet-v1-1-nightly
    Author: Lvmin Zhang (lllyasviel)
"""

class ControlLDM(CustomizedColorizer):
    def __init__(
            self,
            control_stage_config,
            only_mid_control,
            sd_locked = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.sd_locked = sd_locked
        self.model_list.update({"control": self.control_model})


    def on_train_start(self):
        self.dtype = self.first_stage_model.dtype
        self.model.diffusion_model.dtype = self.dtype
        self.control_model.dtype = self.dtype

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

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
        xc = batch[self.cond_stage_key]
        xs = batch[self.control_key]

        x, xs = map(
            lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
            (x, xs)
        )
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
        if self.training:
            uc = [""] * z.shape[0]
            c, uc = self.get_learned_conditioning(xc+uc).detach().chunk(2)
            p = zero_drop(c, self.ucg_rate)
            c = torch.where(p>0, c, uc)
        else:
            c = self.get_learned_conditioning(xc).detach()
        out = [z, dict(c_crossattn=[c], c_concat=[xs])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xs])
        return out


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # cond["c_concat"] = None
        if cond['c_concat'] is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control
            )
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control
            )

        return eps


    def get_trainable_params(self):
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return params


    def switch_to_fp16(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.half()
        unet.middle_block = unet.middle_block.half()
        unet.output_blocks = unet.output_blocks.half()
        unet.dtype = torch.float16

        cunet = self.control_model
        cunet.input_blocks = cunet.input_blocks.half()
        cunet.middle_block = cunet.middle_block.half()
        cunet.middle_block_out = cunet.middle_block_out.half()
        cunet.zero_convs = cunet.zero_convs.half()
        cunet.dtype = torch.float16


    def switch_to_fp32(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.float()
        unet.middle_block = unet.middle_block.float()
        unet.output_blocks = unet.output_blocks.float()
        unet.dtype = torch.float32

        cunet = self.control_model
        cunet.input_blocks = cunet.input_blocks.float()
        cunet.middle_block = cunet.middle_block.float()
        cunet.middle_block_out = cunet.middle_block_out.float()
        cunet.zero_convs = cunet.zero_convs.float()
        cunet.dtype = torch.float32


    @torch.no_grad()
    def log_images(
            self,
            batch,
            N=4,
            n_row=2,
            sample=False,
            step=20,
            ddim_eta=0.0,
            unconditional_guidance_scale=9.0,
            unconditional_guidance_label=None,
            use_ema_scope=True,
            return_inputs=False,
            **kwargs
    ):
        """
            This function is used for batch processing.
            Used with image logger.
        """

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
            z, c, x, xs = out
            log["inputs"] = x
            log["control"] = xs
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
            log["conditioning"] = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key],
                                                 size=x.shape[2] // 25)
        else:
            z, c = out

        crossattn, concat = c["c_crossattn"][0], c["c_concat"][0]
        B, _, H, W = z.shape
        uc_cross = self.get_unconditional_conditioning(B)
        uc_full = {"c_crossattn": [uc_cross], "c_concat": [concat]}
        samples = self.sample(
            cond = c,
            bs = B,
            shape = (self.channels, H, W),
            step = step,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            unconditional_guidance_label = unconditional_guidance_label,
            device = z.device,
        )
        x_samples = self.decode_first_stage(samples.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log


class InferenceWrapper(ControlLDM):
    def prepare_conditions(
            self,
            bs,
            control,
            reference,
            use_local,
            target_texts,
            anchor_texts,
            target_scales,
            enhances,
    ):
        def expand_to_batch_size(bs, x):
            x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
            return x

        crossattn, uc_crossattn = self.get_learned_conditioning([target_texts[0], anchor_texts[0]]).chunk(2)
        control, crossattn, uc_crossattn = map(
            lambda t: expand_to_batch_size(bs, t), (control, crossattn, uc_crossattn)
        )

        null_control = torch.zeros_like(control)
        c = {"c_concat": [control], "c_crossattn": [crossattn]}
        uc = [
            {"c_concat": [control], "c_crossattn": [uc_crossattn]},
            {"c_concat": [null_control], "c_crossattn": [crossattn]}
        ]
        return c, uc


    @torch.no_grad()
    def generate(
            self,
            sampler,
            step: int,
            gs: list[float],
            bs: int,
            cond: dict,
            height: int = 512,
            width: int = 512,
            low_vram: bool = True,
            injection: bool = False,
            injection_cfg: float = 0.5,
            adain: bool = False,
            adain_cfg: float = 0.5,
            hook_xr: torch.Tensor = None,
            hook_sketch: torch.Tensor = None,
            use_local: bool = False,
            target_texts: list[str] = None,
            anchor_texts: list[str] = None,
            target_scales: list[float] = None,
            enhances: list[bool] = None,
            use_rx: bool = False,
            **kwargs,
    ):
        """
            User interface function.
        """

        control = cond["concat"]
        reference = cond["crossattn"]

        if low_vram:
            self.cpu()
            self.low_vram_shift(["cond"] + self.switch_modules)
        else:
            self.low_vram_shift([model for model in self.model_list.keys()])

        c, uc = self.prepare_conditions(
            bs,
            control,
            reference,
            use_local,
            target_texts,
            anchor_texts,
            target_scales,
            enhances,
        )

        if low_vram:
            self.low_vram_shift("first")

        rx = None

        if low_vram:
            self.low_vram_shift(["unet", "control"])

        z = self.sample(
            cond=c,
            uncond=uc,
            bs=bs,
            shape=(self.channels, height // 8, width // 8),
            cfg_scale=gs,
            step=step,
            sampler=sampler,
            x_T=rx
        )

        if low_vram:
            self.low_vram_shift("first")
        return self.decode_first_stage(z.to(self.first_stage_model.dtype))