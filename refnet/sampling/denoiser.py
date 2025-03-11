import torch
import torch.nn as nn

import inspect
import os.path as osp
from typing import Union, Optional
from tqdm import tqdm
from util import load_config
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)


class CFGDenoiser(nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, model, device):
        super().__init__()
        denoiser = CompVisDenoiser if model.parameterization == "eps" else CompVisVDenoiser
        self.model_wrap = denoiser(model, device=device)

    @property
    def inner_model(self):
        return self.model_wrap

    def forward(
            self,
            x,
            sigma,
            cond: dict,
            cond_scale: Union[float, list[float]]
    ):
        """
            Simplify k-diffusion sampler for sketch colorizaiton.
            Available for reference CFG / sketch CFG or Dual CFG
        """
        if not isinstance(cond_scale, list):
            if cond_scale > 1.:
                repeats = 2
            else:
                return self.inner_model(x, sigma, cond=cond)
        else:
            repeats = 3

        x_in = torch.cat([x] * repeats)
        sigma_in = torch.cat([sigma] * repeats)
        x_out = self.inner_model(x_in, sigma_in, cond=cond).chunk(repeats)
        if repeats == 2:
            x_cond, x_uncond = x_out[:]
            return x_uncond + (x_cond - x_uncond) * cond_scale
        else:
            x_cond, x_uncond_0, x_uncond_1 = x_out[:]
            return (x_uncond_0 + (x_cond - x_uncond_0) * cond_scale[0] +
                    x_uncond_1 + (x_cond - x_uncond_1) * cond_scale[1]) * 0.5




scheduler_config_path = "configs/scheduler_cfgs"
class DiffuserDenoiser:
    scheduler_types = {
        "ddim": DDIMScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "pndm": PNDMScheduler,
        "lms": LMSDiscreteScheduler
    }
    def __init__(self, scheduler_type, prediction_type, use_karras=False):
        scheduler_type = scheduler_type.replace("diffuser_", "")
        assert scheduler_type in self.scheduler_types.keys(), "Selected scheduler is not implemented"
        scheduler = self.scheduler_types[scheduler_type]
        scheduler_config = load_config(osp.abspath(osp.join(scheduler_config_path, scheduler_type + ".yaml")))
        if "use_karras_sigmas" in set(inspect.signature(scheduler).parameters.keys()):
            scheduler_config.use_karras_sigmas = use_karras
        self.scheduler = scheduler(prediction_type=prediction_type, **scheduler_config)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def __call__(
            self,
            x,
            cond,
            cond_scale,
            unet,
            timesteps,
            generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
            eta: float = 0.0,
            device: str = "cuda"
    ):
        self.scheduler.set_timesteps(timesteps, device=device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        x = x * self.scheduler.init_noise_sigma
        for i, t in enumerate(tqdm(timesteps)):
            x_t = self.scheduler.scale_model_input(x, t)

            if not isinstance(cond_scale, list):
                if cond_scale > 1.:
                    repeats = 2
                else:
                    repeats = 1
            else:
                repeats = 3

            x_in = torch.cat([x_t] * repeats)
            x_out = unet.apply_model(
                x_in,
                t[None].expand(x_in.shape[0]),
                cond=cond
            )

            if repeats == 1:
                pred = x_out

            elif repeats == 2:
                x_cond, x_uncond = x_out.chunk(2)
                pred = x_uncond + (x_cond - x_uncond) * cond_scale

            else:
                x_cond, x_uncond_0, x_uncond_1 = x_out.chunk(3)
                pred = (x_uncond_0 + (x_cond - x_uncond_0) * cond_scale[0] +
                        x_uncond_1 + (x_cond - x_uncond_1) * cond_scale[1]) * 0.5

            x = self.scheduler.step(
                pred, t, x, **extra_step_kwargs, return_dict=False
            )[0]

        return x