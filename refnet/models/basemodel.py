import torch

from refnet.util import exists, fitting_weights, instantiate_from_config
from refnet.ldm import LatentDiffusion
from util import load_weights, delete_states
from typing import Union
from refnet.sampling import (
    UnetHook,
    torch_dfs,
    KDiffusionSampler,
    DiffuserDenoiser,
)



class GuidanceFlag:
    none = 0
    reference = 1
    sketch = 10
    both = 11


def reconstruct_cond(cond, uncond):
    if not isinstance(uncond, list):
        uncond = [uncond]
    for k in cond.keys():
        for uc in uncond:
            if isinstance(cond[k], list):
                cond[k] = [torch.cat([cond[k][i], uc[k][i]]) for i in range(len(cond[k]))]
            elif isinstance(cond[k], torch.Tensor):
                cond[k] = torch.cat([cond[k], uc[k]])
    return cond


class CustomizedLDM(LatentDiffusion):
    def __init__(
            self,
            dtype = torch.float32,
            sigma_max = None,
            sigma_min = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        self.model_list = {
            "first": self.first_stage_model,
            "cond": self.cond_stage_model,
            "unet": self.model,
        }
        self.switch_cond_modules = ["cond"]
        self.switch_main_modules = ["unet"]
        self.retrieve_attn_modules()
        self.retrieve_attn_layers()

    def init_from_ckpt(
            self,
            path,
            only_model = False,
            logging = False,
            make_it_fit = False,
            ignore_keys: list[str] = (),
    ):
        sd = delete_states(load_weights(path), ignore_keys)
        if make_it_fit:
            sd = fitting_weights(self, sd)

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model \
            else self.model.load_state_dict(sd, strict=False)

        filtered_missing = []
        filtered_unexpect = []
        for k in missing:
            if not k.find("cond_stage_model") > -1 and not k.find("img_embedder") > -1 and not k.find("lora") > -1:
                filtered_missing.append(k)
        for k in unexpected:
            if not k.find("cond_stage_model") > -1 and not k.find("img_embedder") > -1:
                filtered_unexpect.append(k)

        print(
            f"Restored from {path} with {len(filtered_missing)} filtered missing and "
            f"{len(filtered_unexpect)} filtered unexpected keys")
        if logging:
            if len(missing) > 0:
                print(f"Filtered missing Keys: {filtered_missing}")
            if len(unexpected) > 0:
                print(f"Filtered unexpected Keys: {filtered_unexpect}")


    def sample(
            self,
            cond: dict,
            uncond: Union[dict, list[dict]] = None,
            cfg_scale: Union[float, list[float]] = 1.,
            bs: int = 1,
            shape: Union[tuple, list] = None,
            step: int = 20,
            sampler = "DPM++ 3M SDE",
            scheduler = "Automatic",
            device = "cuda",
            x_T = None,
            seed = None,
            deterministic = False,
            **kwargs
    ):
        shape = shape or (self.channels, self.image_size, self.image_size)
        x = x_T or torch.randn(bs, *shape, device=device)

        if exists(uncond):
            cond = reconstruct_cond(cond, uncond)

        if sampler.startswith("diffuser"):
            # Using huggingface diffuser noise sampler and scheduler
            sampler = DiffuserDenoiser(
                sampler,
                prediction_type = "v_prediction" if self.parameterization == "v" else "epsilon",
                use_karras = scheduler == "Karras"
            )

            samples = sampler(
                x,
                cond,
                cond_scale=cfg_scale,
                unet=self,
                timesteps=step,
                generator=torch.manual_seed(seed) if exists(seed) else None,
                device=device
            )

        else:
            # Using k-diffusion sampler and noise scheduler
            seed = seed or torch.seed()
            sampler = KDiffusionSampler(sampler, scheduler, self, device)
            sigmas = sampler.get_sigmas(step)
            extra_args = {
                "cond": cond,
                "cond_scale": cfg_scale,
            }
            seed = [seed for _ in range(bs)] if deterministic else seed
            samples = sampler(x, sigmas, extra_args, seed, deterministic, step)

        return samples

    def switch_to_fp16(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.to(self.half_precision_dtype)
        unet.middle_block = unet.middle_block.to(self.half_precision_dtype)
        unet.output_blocks = unet.output_blocks.to(self.half_precision_dtype)
        self.dtype = self.half_precision_dtype
        unet.dtype = self.half_precision_dtype

    def switch_to_fp32(self):
        unet = self.model.diffusion_model
        unet.input_blocks = unet.input_blocks.float()
        unet.middle_block = unet.middle_block.float()
        unet.output_blocks = unet.output_blocks.float()
        self.dtype = torch.float32
        unet.dtype = torch.float32

    def switch_vae_to_fp16(self):
        self.first_stage_model = self.first_stage_model.to(self.half_precision_dtype)

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
        scale_factor_levels = {"high": 1., "middle": 0.5, "low": 0.25, "bottom": 0.125}

        from refnet.modules.transformer import BasicTransformerBlock
        self.attn_modules = {
            "high": [0, 1, 13, 14, 15],
            "middle": [2, 3, 10, 11, 12],
            "low": [4, 5, 7, 8, 9],
            "bottom": [6],
            "encoder": [0, 1, 2, 3, 4, 5]
        }

        attn_modules = []
        for module in torch_dfs(self.model.diffusion_model):
            if isinstance(module, BasicTransformerBlock):
                attn_modules.append(module)

        self.attn_modules["modules"] = attn_modules

        for k in ["high", "middle", "low", "bottom"]:
            scale_factor = scale_factor_levels[k]
            for attn in self.attn_modules[k]:
                attn_modules[attn].scale_factor = scale_factor


    def retrieve_attn_layers(self):
        self.attn_layers = []
        for module in (self.attn_modules["modules"]):
            self.attn_layers.append(module.attn2)


class CustomizedColorizer(CustomizedLDM):
    def __init__(
            self,
            control_encoder_config,
            proj_config,
            token_type = "local",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_encoder = instantiate_from_config(control_encoder_config)
        self.proj = instantiate_from_config(proj_config)
        self.token_type = token_type
        self.model_list.update({"control_encoder": self.control_encoder, "proj": self.proj})
        self.switch_cond_modules += ["control_encoder", "proj"]


    def switch_to_fp16(self):
        self.control_encoder = self.control_encoder.to(self.half_precision_dtype)
        super().switch_to_fp16()


    def switch_to_fp32(self):
        self.control_encoder = self.control_encoder.float()
        super().switch_to_fp32()


from refnet.modules.unet import hack_inference_forward
class CustomizedWrapper():
    def __init__(self):
        self.scaling_sample = False
        self.guidance_steps = (0, 1)
        self.no_guidance_steps = (-0.05, 0.05)
        hack_inference_forward(self.model.diffusion_model)

    def adjust_reference_scale(self, scale_kwargs):
        if isinstance(scale_kwargs, dict):
            if scale_kwargs["level_control"]:
                for key in scale_kwargs["scales"]:
                    for idx in self.attn_modules[key]:
                        self.attn_modules["modules"][idx].reference_scale = scale_kwargs["scales"][key]
            else:
                for idx, s in enumerate(scale_kwargs["scales"]):
                    self.attn_modules["modules"][idx].reference_scale = s
        else:
            for module in self.attn_modules["modules"]:
                module.reference_scale = scale_kwargs

    def adjust_fgbg_scale(self, fg_scale, bg_scale, merge_scale, mask_threshold):
        for layer in self.attn_layers:
            layer.fg_scale = fg_scale
            layer.bg_scale = bg_scale
            layer.merge_scale = merge_scale
            layer.mask_threshold = mask_threshold
        # for layer in self.attn_modules["modules"]:
        #     layer.fg_scale = fg_scale
        #     layer.bg_scale = bg_scale
        #     layer.merge_scale = merge_scale
        #     layer.mask_threshold = mask_threshold

    def apply_model(self, x_noisy, t, cond):
        tr = 1 - t[0] / (self.num_timesteps - 1)
        crossattn = cond["context"][0]
        if ((tr < self.guidance_steps[0] or tr > self.guidance_steps[1]) or
                (tr >= self.no_guidance_steps[0] and tr <= self.no_guidance_steps[1])):
            crossattn = torch.zeros_like(crossattn)[:, :1]
        cond["context"] = [crossattn]
        return self.model(x_noisy, t, **cond)


    def prepare_conditions(self, *args, **kwargs):
        raise NotImplementedError("Inputs preprocessing function is not implemented.")


    def check_manipulate(self, scales):
        if exists(scales) and len(scales) > 0:
            for scale in scales:
                if scale > 0:
                    return True
        return False

    @torch.no_grad()
    def generate(
            self,
            # Conditional inputs
            cond: dict,
            ctl_scale: float,
            merge_scale: float,
            mask_scale: float,
            mask_thresh: float,
            mask_thresh_sketch: float,

            # Sampling settings
            sampler,
            scheduler,
            step: int,
            bs: int,
            gs: list[float],
            strength: Union[float, list[float]],
            fg_strength: float,
            bg_strength: float,
            seed: int,
            start_step: float = 0.0,
            end_step: float = 1.0,
            no_start_step: float = -0.05,
            no_end_step: float = -0.05,
            deterministic: bool = False,
            style_enhance: bool = False,
            bg_enhance: bool = False,
            fg_enhance: bool = False,
            geometry_map: bool = False,
            height: int = 512,
            width: int = 512,

            # Injection settings
            injection: bool = False,
            injection_cfg: float = 0.5,
            injection_control: float = 0,
            injection_start_step: float = 0,
            hook_xr: torch.Tensor = None,
            # hook_xs: torch.Tensor = None,

            # Additional settings
            low_vram: bool = True,
            return_intermediate = False,
            manipulation_params = None,
            **kwargs,
    ):
        """
            User interface function.
        """
        hook_unet = UnetHook()

        self.guidance_steps = (start_step, end_step)
        self.no_guidance_steps = (no_start_step, no_end_step)
        self.adjust_reference_scale(strength)
        self.adjust_fgbg_scale(fg_strength, bg_strength, merge_scale, mask_thresh_sketch)

        if low_vram:
            self.low_vram_shift(self.switch_cond_modules)
        else:
            self.low_vram_shift(list(self.model_list.keys()))

        c, uc = self.prepare_conditions(
            bs = bs,
            control_scale = ctl_scale,
            merge_scale = merge_scale,
            mask_scale = mask_scale,
            mask_threshold_ref = mask_thresh,
            mask_threshold_sketch = mask_thresh_sketch,
            style_enhance = style_enhance,
            bg_enhance = bg_enhance,
            fg_enhance = fg_enhance,
            geometry_map = geometry_map,
            height = height,
            width = width,
            bg_strength = bg_strength,
            **cond,
            **manipulation_params,
            **kwargs
        )
        if low_vram:
            for k in cond:
                del cond[k]
            torch.cuda.empty_cache()

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
            rx = self.get_first_stage_encoding(hook_xr.to(self.first_stage_model.dtype))
            hook_unet.enhance_reference(
                model = self.model,
                ldm = self,
                bs = bs * repeat,
                s = -hook_xr.to(self.dtype),
                r = rx,
                style_cfg = injection_cfg,
                control_cfg = injection_control,
                gr_indice = gr_indice,
                start_step = injection_start_step,
            )

        if low_vram:
            self.low_vram_shift(self.switch_main_modules)

        z = self.sample(
            cond = c,
            uncond = uc,
            bs = bs,
            shape = (self.channels, height // 8, width // 8),
            cfg_scale = gs,
            step = step,
            sampler = sampler,
            scheduler = scheduler,
            seed = seed,
            deterministic = deterministic,
            return_intermediate = return_intermediate,
        )

        if injection:
            hook_unet.restore(self.model)

        if low_vram:
            self.low_vram_shift("first")
        return self.decode_first_stage(z.to(self.first_stage_model.dtype))