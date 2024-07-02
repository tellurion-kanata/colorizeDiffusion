import torch

from refnet.util import zero_drop, exists, default, append_dims, fitting_weights
from refnet.sampling import UnetHook, CFGDenoiser, torch_dfs
from ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from sgm.util import instantiate_from_config
from util import load_weights, delete_states
from typing import Union
from tqdm import tqdm
from k_diffusion import (
    create_noise_sampler,
    get_sigmas_karras,
    get_sigmas_exponential,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
)


SAMPLER_FUNC_DICT = {
    # "lms": {"sampler": sample_lms},
    # "euler": {"sampler": sample_euler},

    # "dpm_2m": {"sampler": sample_dpmpp_2m},
    "ddim": {"scheduler": None},
    "dpm_vp": {"scheduler": None},
    "dpm_2m_sde": {"sampler": sample_dpmpp_2m_sde},
    "dpm_2m_sde_exponential": {"scheduler": get_sigmas_exponential, "sampler": sample_dpmpp_2m_sde},
    "dpm_2m_sde_karras": {"scheduler": get_sigmas_karras, "sampler": sample_dpmpp_2m_sde},

    "dpm_3m_sde": {"sampler": sample_dpmpp_3m_sde, "discard_next_to_last_sigma": True},
    "dpm_3m_sde_exponential": {"scheduler": get_sigmas_exponential, "sampler": sample_dpmpp_3m_sde, "discard_next_to_last_sigma": True},
    "dpm_3m_sde_karras": {"scheduler": get_sigmas_karras, "sampler": sample_dpmpp_3m_sde, "discard_next_to_last_sigma": True},
}

class GuidanceFlag:
    none = 0
    reference = 1
    sketch = 10
    both = 11

def get_sampler_list():
    return [key for key in SAMPLER_FUNC_DICT.keys()]


def reconstruct_cond(cond, uncond):
    if not isinstance(uncond, list):
        uncond = [uncond]
    for k in cond.keys():
        for uc in uncond:
            if isinstance(cond[k], list):
                cond[k] = [torch.cat([cond[k][i], uc[k][i]]) for i in range(len(cond[k]))]
            else:
                cond[k] = [torch.cat(cond[k], uc[k])]
    return cond


class CustomizedLDM(LatentDiffusion):
    def __init__(
            self,
            ucg_rate = 0.,
            noisy_training = False,
            offset_noise_level = 0.,
            ucg_range = 0.,
            dtype = torch.float32,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.ucg_rate = ucg_rate
        self.ucg_range = ucg_range
        self.noisy_training = noisy_training
        self.offset_noise_level = offset_noise_level

        self.model_list = {
            "first": self.first_stage_model,
            "cond": self.cond_stage_model,
            "unet": self.model,
        }
        self.switch_cond_modules = []
        self.switch_main_modules = []
        self.retrieve_attn_modules()


    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, logging=False, make_it_fit=False):
        sd = delete_states(load_weights(path), ignore_keys)
        if make_it_fit:
            sd = fitting_weights(self, sd)

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model \
            else self.model.load_state_dict(sd, strict=False)

        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if logging:
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")

    def sample(
            self,
            cond: dict,
            uncond: Union[dict, list[dict]] = None,
            cfg_scale: Union[float, list[float]] = 1.,
            bs: int = 1,
            shape: Union[tuple, list] = None,
            step: int = 20,
            sampler = "dpm_3m_sde_karras",
            device = "cuda",
            x_T = None,
            seed = None,
            deterministic = False,
            return_intermediate = False,
            **kwargs
    ):
        shape = shape if exists(shape) else (self.channels, self.image_size, self.image_size)
        x = x_T if exists(x_T) else torch.randn(bs, *shape, device=device)
        denoiser = SAMPLER_FUNC_DICT[sampler]

        if sampler.find("sde") > -1:
            if exists(uncond):
                cond = reconstruct_cond(cond, uncond)
            seed = seed if exists(seed) else torch.seed()
            sigmas_sampler = denoiser.get("scheduler", None)
            sampler_func = denoiser["sampler"]
            discard = denoiser.get("discard_next_to_last_sigma", False)

            self.alphas_cumprod = self.alphas_cumprod.to(device)
            model_wrapper = CFGDenoiser(self)
            step += 1 if discard else 0
            sigmas = sigmas_sampler(
                step,
                sigma_min = model_wrapper.inner_model.sigma_min,
                sigma_max = model_wrapper.inner_model.sigma_max,
                device = device,
            ) if exists(sigmas_sampler) else model_wrapper.inner_model.get_sigmas(step)
            if discard:
                sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

            extra_args = {
                "cond": cond,
                "cond_scale": cfg_scale,
            }
            if deterministic:
                seed = [seed for _ in range(bs)]
            if sampler.find("sde") > -1:
                samples = sampler_func(model_wrapper, x, sigmas, extra_args,
                                       noise_sampler=create_noise_sampler(x, sigmas, seed))
            else:
                samples = sampler_func(model_wrapper, x, sigmas, extra_args)
        else:
            if sampler == "dpm_vp":
                sampler = DPMSolverSampler(self)
            elif sampler == "ddim":
                sampler = DDIMSampler(self)
            else:
                sampler = self
                sampler.sample = self.p_sample_loop

            samples, intermediates = sampler.sample(
                step,
                bs,
                shape,
                cond,
                unconditional_conditioning=uncond,
                x_T=x,
                unconditional_guidance_scale=cfg_scale,
                verbose=False,
                log_every_t = step // 5 if return_intermediate else self.num_timesteps,
                **kwargs
            )
            if return_intermediate:
                intermediates["pred_x0"].append(samples)
                samples = torch.cat(intermediates["x_inter"]+intermediates["pred_x0"], 0)

        return samples


    def on_train_start(self):
        self.dtype = self.first_stage_model.dtype
        self.model.diffusion_model.dtype = self.dtype

    def timedepend_preprocess(self, cond, t):
        # t = (1 - (t/(self.num_timesteps-1)) ** 3.) * (self.num_timesteps - 1)
        # t = t.long()
        crossattn = cond["c_crossattn"][0]
        if self.noisy_training:
            crossattn = self.q_sample(x_start=crossattn, t=t).to(self.dtype)

        if self.ucg_rate > 0:
            crossattn = zero_drop(
                crossattn,
                # self.ucg_rate + self.ucg_range * t / (self.num_timesteps - 1)
                self.ucg_rate
            ) * crossattn
        cond["c_crossattn"] = [crossattn]
        return cond, t

    def p_losses(self, x_start, cond, t, noise=None):
        cond, t = self.timedepend_preprocess(cond, t)

        # loss calculation is refined according to sdxl
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.offset_noise_level > 0.:
            noise += self.offset_noise_level * append_dims(
                torch.randn(x_start.shape[0], device=x_start.device), x_start.ndim
            )

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise).to(self.dtype)
        w = append_dims(self.sqrt_recipm1_alphas_cumprod[t] ** -2, x_start.ndim)
        model_output = self.predict_start_from_noise(
            x_t = x_noisy,
            t = t,
            noise = self.apply_model(x_noisy, t, cond)
        )

        loss = torch.mean(w * (model_output - x_start) ** 2)
        return loss

    def get_trainable_params(self):
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            tqdm.write(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params += self.cond_stage_model.parameters()
        return params

    def switch_to_fp16(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.half()
        unet.middle_block = unet.middle_block.half()
        unet.output_blocks = unet.output_blocks.half()
        self.dtype = torch.float16
        unet.dtype = torch.float16

    def switch_to_fp32(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.float()
        unet.middle_block = unet.middle_block.float()
        unet.output_blocks = unet.output_blocks.float()
        self.dtype = torch.float32
        unet.dtype = torch.float32

    def switch_vae_to_fp16(self):
        self.first_stage_model = self.first_stage_model.half()

    def switch_vae_to_fp32(self):
        self.first_stage_model = self.first_stage_model.float()

    def low_vram_shift(self, cuda_list: Union[str, list[str]]):
        if not isinstance(cuda_list, list):
            cuda_list = [cuda_list]

        cpu_list = self.model_list.keys() - cuda_list
        for model in cpu_list:
            self.model_list[model] = self.model_list[model].cpu()
        torch.cuda.empty_cache()

        for model in cuda_list:
            self.model_list[model] = self.model_list[model].cuda()


    def retrieve_attn_modules(self):
        from ldm.modules.attention import BasicTransformerBlock
        from refnet.modules.attention import IPTransformerBlock
        if self.model.diffusion_model.only_decoder:
            self.attn_modules = {
                "high": [7, 8, 9],
                "middle": [4, 5, 6],
                "low": [1, 2, 3]
            }
        else:
            self.attn_modules = {
                "high": [0, 1, 13, 14, 15],
                "middle": [2, 3, 10, 11, 12],
                "low": [4, 5, 7, 8, 9],
                "encoder": [0, 1, 2, 3, 4, 5]
            }

        attn_modules = []
        for module in torch_dfs(self.model.diffusion_model):
            if isinstance(module, BasicTransformerBlock) or isinstance(module, IPTransformerBlock):
                attn_modules.append(module)
        self.attn_modules["modules"] = attn_modules


class CustomizedColorizer(CustomizedLDM):
    def __init__(
            self,
            control_encoder_config,
            control_key,
            control_drop = 0.0,
            warp_p = 0.,
            token_type = "local",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_encoder = instantiate_from_config(control_encoder_config)
        self.control_key = control_key
        self.control_drop = control_drop
        self.token_type = token_type
        self.warp_p = warp_p
        self.model_list.update({"control_encoder": self.control_encoder})

    def get_trainable_params(self):
        return list(self.model.parameters()) + list(self.control_encoder.parameters())

    def timedepend_preprocess(self, cond, t):
        if self.control_drop > 0:
            concat = cond["c_concat"][0]
            ones = torch.ones_like(concat)
            drop = zero_drop(concat, self.control_drop) * ones
            concat = torch.where(drop > 0, concat, -ones)
            cond["c_concat"] = [concat]
        return super().timedepend_preprocess(cond, t)


    def switch_to_fp16(self):
        self.control_encoder = self.control_encoder.half()
        super().switch_to_fp16()


    def switch_to_fp32(self):
        self.control_encoder = self.control_encoder.float()
        super().switch_to_fp32()


    def get_unconditional_conditioning(self, x, null_label=None):
        return torch.zeros_like(x)

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
        with torch.no_grad():
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]
            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key]

            x, xc, xs = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs)
            )
            c = self.cond_stage_model.encode(xc, self.token_type, warp_p=self.warp_p).detach()
            z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        concat = self.control_encoder(xs)
        out = [z, dict(c_crossattn=[c.to(self.dtype)], c_concat=concat)]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs])
        return out

    @torch.no_grad()
    def log_images(
            self,
            batch,
            N=4,
            sampler="dpm_vp",
            step=20,
            unconditional_guidance_scale=9.0,
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
            z, c, x, xc, xs = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
        else:
            z, c = out

        crossattn, concat = c["c_crossattn"][0], c["c_concat"]
        B, _, H, W = z.shape
        uc_cross = self.get_unconditional_conditioning(crossattn)
        uc_full = {"c_crossattn": [uc_cross], "c_concat": concat}
        samples = self.sample(
            cond = c,
            bs = B,
            shape = (self.channels, H, W),
            step = step,
            sampler = sampler,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            device = z.device,
        )
        x_samples = self.decode_first_stage(samples.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log



class CustomizedWrapper():
    def __init__(self):
        self.scaling_sample = False
        self.guidance_steps = (0, 1)
        self.no_guidance_steps = (-0.05, 0.05)


    def adjust_reference_scale(self, scale_kwargs):
        if isinstance(scale_kwargs, dict):
            if scale_kwargs["level_control"]:
                for key in scale_kwargs["scales"]:
                    for idx in self.attn_modules[key]:
                        self.attn_modules["modules"][idx].reference_scale = scale_kwargs["scales"][key]
            else:
                if self.model.diffusion_model.only_decoder:
                    scale_kwargs["scales"] = scale_kwargs["scales"][7:]
                for idx, s in enumerate(scale_kwargs["scales"]):
                    self.attn_modules["modules"][idx].reference_scale = s
        else:
            for module in self.attn_modules["modules"]:
                module.reference_scale = scale_kwargs


    def apply_model(self, x_noisy, t, cond, return_ids=False):
        tr = 1 - t[0] / (self.num_timesteps - 1)
        concat, crossattn = cond["c_concat"], cond["c_crossattn"][0]
        if ((tr < self.guidance_steps[0] or tr > self.guidance_steps[1]) or
                (tr >= self.no_guidance_steps[0] and tr <= self.no_guidance_steps[1])):
            crossattn = torch.zeros_like(crossattn)[:, :1]
        return self.model(x_noisy, t, c_concat=concat, c_crossattn=[crossattn], c_adm=cond.get("c_adm", None))


    def prepare_conditions(self, *args, **kwargs):
        raise NotImplementedError("Inputs preprocessing function is not implemented.")


    def check_manipulate(self, scales):
        if exists(scales) and len(scales) > 0:
            for scale in scales:
                if scale > 0:
                    return True
        return False


    def adjust_control_scale(self, scale):
        self.model.diffusion_model.control_scale = scale


    @torch.no_grad()
    def generate(
            self,
            sampler,
            step: int,
            gs: list[float],
            ctl_scale: float,
            strength: Union[float, list[float]],
            bs: int,
            cond: dict,
            seed: int,
            height: int = 512,
            width: int = 512,
            low_vram: bool = True,
            injection: bool = False,
            injection_cfg: float = 0.5,
            injection_control: float = 0,
            injection_start_step: float = 0,
            hook_xr: torch.Tensor = None,
            hook_xs: torch.Tensor = None,
            use_local: bool = False,
            start_step: float = 0.0,
            end_step: float = 1.0,
            no_start_step: float=-0.05,
            no_end_step: float=-0.05,
            manipulation_params = None,
            return_intermediate = False,
            **kwargs,
    ):
        """
            User interface function.
        """
        hook_unet = UnetHook()
        control = cond.get("concat", None)
        reference = cond.get("crossattn", None)
        self.guidance_steps = (start_step, end_step)
        self.no_guidance_steps = (no_start_step, no_end_step)
        self.adjust_reference_scale(strength)
        self.adjust_control_scale(ctl_scale)

        if low_vram:
            self.low_vram_shift(["cond", "control_encoder"] + self.switch_cond_modules)
        else:
            self.low_vram_shift(list(self.model_list.keys()))

        c, uc = self.prepare_conditions(
            bs = bs,
            sketch = control,
            reference = reference,
            use_local = use_local,
            **manipulation_params,
        )

        cfg = int(gs[0] > 1) * GuidanceFlag.reference + int(gs[1] > 1) * GuidanceFlag.sketch
        gr_indice = [] if (cfg == GuidanceFlag.none or cfg == GuidanceFlag.sketch) else [i for i in range(bs, bs*2)]
        repeat = 1
        if cfg == GuidanceFlag.none:
            gs = 1
            uc = None
        if cfg == GuidanceFlag.reference:
            gs = gs[0]
            uc = uc[0]
            repeat = 2
        if cfg == GuidanceFlag.sketch:
            gs = gs[1]
            uc = uc[1]
            repeat = 2
        if cfg == GuidanceFlag.both:
            repeat = 3

        if low_vram:
            self.low_vram_shift("first")

        if injection:
            rx = self.get_first_stage_encoding(self.encode_first_stage(hook_xr.to(self.first_stage_model.dtype)))
            hook_unet.enhance_reference(
                model = self.model,
                ldm = self,
                bs = bs * repeat,
                s = -hook_xr.to(self.dtype),
                r = rx,
                injection = injection,
                style_cfg = injection_cfg,
                control_cfg = injection_control,
                gr_indice = gr_indice,
                start_step = injection_start_step,
            )

        if low_vram:
            self.low_vram_shift(["unet"] + self.switch_main_modules)

        z = self.sample(
            cond = c,
            uncond = uc,
            bs = bs,
            shape = (self.channels, height // 8, width // 8),
            cfg_scale = gs,
            step = step,
            sampler = sampler,
            seed = seed,
            return_intermediate = return_intermediate,
        )

        if injection:
            hook_unet.restore(self.model)

        if low_vram:
            self.low_vram_shift("first")
        return self.decode_first_stage(z.to(self.first_stage_model.dtype))