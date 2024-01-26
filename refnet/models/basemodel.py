import torch
import itertools

from refnet.sampling import UnetHook, CFGDenoiser
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from sgm.util import default, append_dims, exists
from util import load_weights, delete_states
from typing import Union
from tqdm import tqdm
from k_diffusion.sampling import (
    get_sigmas_karras,
    get_sigmas_exponential,
    sample_lms,
    sample_euler,
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
)


SAMPLER_FUNC_DICT = {
    # "lms": {"sampler": sample_lms},
    # "euler": {"sampler": sample_euler},

    # "dpm_2m": {"sampler": sample_dpmpp_2m},
    "dpm_vp": {"sceduler": None},
    "dpm_2m_karras": {"scheduler": get_sigmas_karras, "sampler": sample_dpmpp_2m},
    "dpm_2m_exponential": {"scheduler": get_sigmas_exponential, "sampler": sample_dpmpp_2m},

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


def zero_drop(x, p):
    return torch.bernoulli((1 - p) * append_dims(torch.ones(x.shape[0], device=x.device, dtype=x.dtype), x.ndim))


def get_sampler_list():
    return [key for key in SAMPLER_FUNC_DICT.keys()]


def fitting_weights(model, sd):
    n_params = len([name for name, _ in
                    itertools.chain(model.named_parameters(),
                                    model.named_buffers())])
    for name, param in tqdm(
            itertools.chain(model.named_parameters(),
                            model.named_buffers()),
            desc="Fitting old weights to new weights",
            total=n_params
    ):
        if not name in sd:
            continue
        old_shape = sd[name].shape
        new_shape = param.shape
        assert len(old_shape) == len(new_shape)
        if len(new_shape) > 2:
            # we only modify first two axes
            assert new_shape[2:] == old_shape[2:]
        # assumes first axis corresponds to output dim
        if not new_shape == old_shape:
            new_param = param.clone()
            old_param = sd[name]
            if len(new_shape) == 1:
                for i in range(new_param.shape[0]):
                    new_param[i] = old_param[i % old_shape[0]]
            elif len(new_shape) >= 2:
                for i in range(new_param.shape[0]):
                    for j in range(new_param.shape[1]):
                        new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                n_used_old = torch.ones(old_shape[1])
                for j in range(new_param.shape[1]):
                    n_used_old[j % old_shape[1]] += 1
                n_used_new = torch.zeros(new_shape[1])
                for j in range(new_param.shape[1]):
                    n_used_new[j] = n_used_old[j % old_shape[1]]

                n_used_new = n_used_new[None, :]
                while len(n_used_new.shape) < len(new_shape):
                    n_used_new = n_used_new.unsqueeze(-1)
                new_param /= n_used_new

            sd[name] = new_param
    return sd


class CustomizedLDM(LatentDiffusion):
    def __init__(
            self,
            control_key,
            ucg_rate = 0.,
            offset_noise_level = 0.,
            emb_offset_noise_level = 0.,
            noisy_training = False,
            dtype = torch.float32,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.ucg_rate = ucg_rate
        self.control_key = control_key
        self.noisy_training = noisy_training
        self.offset_noise_level = offset_noise_level
        self.emb_offset_noise_level = emb_offset_noise_level

        self.model_list = {
            "first": self.first_stage_model,
            "cond": self.cond_stage_model,
            "unet": self.model,
        }
        self.switch_modules = []


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
            **kwargs
    ):
        shape = shape if exists(shape) else (self.channels, self.image_size, self.image_size)
        randn = x_T if exists(x_T) else torch.randn(bs, *shape, device=device)

        if sampler != "dpm_vp":
            denoiser = SAMPLER_FUNC_DICT[sampler]
            sigmas_sampler = denoiser.get("scheduler", None)
            sampler = denoiser["sampler"]
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
                "uncond": uncond,
                "cond_scale": cfg_scale,
            }
            return sampler(model_wrapper, randn, sigmas, extra_args)

        else:
            sampler = DPMSolverSampler(self)
            samples, intermediates = sampler.sample(
                step,
                bs,
                shape,
                cond,
                unconditional_conditioning=uncond,
                x_T=x_T,
                unconditional_guidance_scale=cfg_scale,
                verbose=False,
                **kwargs
            )
            return samples

    def on_train_start(self):
        self.dtype = self.first_stage_model.dtype
        self.model.diffusion_model.dtype = self.dtype

    def p_losses(self, x_start, cond, t, noise=None):
        # add noise to reference embedding & drop reference condition here.
        crossattn = cond["c_crossattn"][0]
        if self.noisy_training:
            crossattn = self.q_sample(x_start=crossattn, t=t).to(self.dtype)
            crossattn += self.emb_offset_noise_level * append_dims(
                torch.randn(crossattn.shape[0], device=crossattn.device), crossattn.ndim
            )

        crossattn = zero_drop(crossattn, self.ucg_rate) * crossattn
        cond["c_crossattn"]  = [crossattn]

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

    def get_unconditional_conditioning(self, c, null_label=None):
        return torch.zeros_like(c, device=c.device).to(c.dtype)

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
        unet.dtype = torch.float16

    def switch_to_fp32(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.float()
        unet.middle_block = unet.middle_block.float()
        unet.output_blocks = unet.output_blocks.float()
        unet.dtype = torch.float32

    def switch_vae_to_fp16(self):
        self.first_stage_model = self.first_stage_model.half()

    def switch_vae_to_fp32(self):
        self.first_stage_model = self.first_stage_model.float()

    def low_vram_shift(self, cuda_list: Union[str, list[str]]):
        if not isinstance(cuda_list, list):
            cuda_list = [cuda_list]

        cpu_list = self.model_list.keys() - cuda_list
        for model in cuda_list:
            self.model_list[model] = self.model_list[model].cuda()
        for model in cpu_list:
            self.model_list[model] = self.model_list[model].cpu()


class CustomizedColorizer(CustomizedLDM):
    @torch.no_grad()
    def log_images(
            self,
            batch,
            N=4,
            n_row=2,
            sampler="dpm_vp",
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
            z, c, x, xc, xs = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
        else:
            z, c = out

        crossattn, concat = c["c_crossattn"][0], c["c_concat"][0]
        B, _, H, W = z.shape
        uc_cross = self.get_unconditional_conditioning(crossattn)
        uc_full = {"c_crossattn": [uc_cross], "c_concat": [concat]}
        samples = self.sample(
            cond = c,
            bs = B,
            shape = (self.channels, H, W),
            step = step,
            sampler = sampler,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            unconditional_guidance_label = unconditional_guidance_label,
            device = z.device,
        )
        x_samples = self.decode_first_stage(samples.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log


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
            use_rx: bool = False,
            manipulation_params = None,
            **kwargs,
    ):
        """
            User interface function.
        """

        hook_unet = UnetHook()
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
            **manipulation_params
        )

        cfg = int(gs[0] > 1) * GuidanceFlag.reference + int(gs[1] > 1) * GuidanceFlag.sketch
        gr_indice = None if (cfg == GuidanceFlag.none or cfg == GuidanceFlag.sketch) else 1
        if cfg == GuidanceFlag.none:
            gs = 1
        if cfg == GuidanceFlag.reference:
            gs = gs[0]
            uc = uc[0]
        if cfg == GuidanceFlag.sketch:
            gs = gs[1]
            uc = uc[1]

        if low_vram:
            self.low_vram_shift("first")

        rx = None
        if injection or adain:
            rx = self.get_first_stage_encoding(self.encode_first_stage(hook_xr.to(self.first_stage_model.dtype)))
            hook_unet.enhance_reference(
                model=self.model,
                ldm=self,
                s=hook_sketch,
                r=rx,
                injection=injection,
                style_cfg=injection_cfg,
                adain=adain,
                gn_weight=adain_cfg,
                gr_indice=gr_indice,
            )

        if use_rx:
            t = torch.ones((bs,)).long() * (self.num_timesteps - 1)
            if not exists(rx):
                rx = self.get_first_stage_encoding(self.encode_first_stage(hook_xr.to(self.first_stage_model.dtype)))
            rx = self.q_sample(rx.cpu(), t).cuda()
        else:
            rx = None

        if low_vram:
            self.low_vram_shift("unet")

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

        if injection or adain:
            hook_unet.restore(self.model)

        if low_vram:
            self.low_vram_shift("first")
        return self.decode_first_stage(z.to(self.first_stage_model.dtype))


    def switch_to_fp16(self):
        self.model.diffusion_model.semantic_input_blocks = self.model.diffusion_model.semantic_input_blocks.half()
        super().switch_to_fp16()


    def switch_to_fp32(self):
        self.model.diffusion_model.semantic_input_blocks = self.model.diffusion_model.semantic_input_blocks.float()
        super().switch_to_fp32()

    def check_manipulate(self, scales):
        if exists(scales) and len(scales) > 0:
            for scale in scales:
                if scale > 0:
                    return True
        return False