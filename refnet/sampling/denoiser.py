import torch
import torch.nn as nn

from k_diffusion.external import CompVisDenoiser
from typing import Union

def reconstruct_cond(cond_dict, uncond_dict):
    if not isinstance(uncond_dict, list):
        uncond_dict = [uncond_dict]
    new_dict = cond_dict.copy()
    for key in cond_dict.keys():
        for uc in uncond_dict:
            new_dict[key] = [torch.cat(new_dict[key] + uc[key], 0)]
    return new_dict


class CFGDenoiser(nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, model):
        super().__init__()
        self.model_wrap = CompVisDenoiser(model)

    @property
    def inner_model(self):
        return self.model_wrap

    def forward(
            self,
            x,
            sigma,
            cond: dict,
            uncond: Union[dict, list[dict]],
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

        combined_cond = reconstruct_cond(cond, uncond)

        x_in = torch.cat([x] * repeats)
        sigma_in = torch.cat([sigma] * repeats)
        x_out = self.inner_model(x_in, sigma_in, cond=combined_cond).chunk(repeats)
        if repeats == 2:
            x_cond, x_uncond = x_out[:]
            return x_uncond + (x_cond - x_uncond) * cond_scale
        else:
            x_cond, x_uncond_0, x_uncond_1 = x_out[:]
            return (x_uncond_0 + (x_cond - x_uncond_0) * cond_scale[0] +
                    x_uncond_1 + (x_cond - x_uncond_1) * cond_scale[1]) * 0.5