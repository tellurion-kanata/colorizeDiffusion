import torch

from refnet.models.basemodel import CustomizedColorizer, UnetHook, GuidanceFlag
from sgm.util import exists


"""
    This class is for Colorize Diffusion v1 models. 
"""

def latent_shuffle(x: torch.Tensor):
    b, n, c = x.shape

    new_ind = torch.randperm(n)
    shuffled = x[:, new_ind]
    return shuffled.contiguous()


class ColorizeDiffusion(CustomizedColorizer):
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
        with torch.no_grad():
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]

            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key][:, :1]

            x, xc, xs = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs)
            )
            z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            c = self.cond_stage_model.encode(xc, "local").detach()

            if self.training:
                c = latent_shuffle(c)

        out = [z, dict(c_crossattn=[c], c_concat=[xs])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs])
        return out

from refnet.sampling.manipulation import local_manipulate, get_heatmaps

class InferenceWrapper(ColorizeDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = 0


    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if 1 - (t[0] / (self.num_timesteps + 1e-4)) < self.noise_level:
            crossattn = cond["c_crossattn"][0]
            cond["c_crossattn"] = [self.q_sample(crossattn.cpu(), t.long().cpu()).cuda()]
        return super().apply_model(x_noisy, t, cond, return_ids)


    def prepare_conditions(
            self,
            bs,
            control,
            reference,
            targets,
            anchors,
            controls,
            target_scales,
            enhances,
            thresholds_list,
            *args,
            **kwargs
    ):
        def expand_to_batch_size(bs, x):
            x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
            return x

        manipulate = self.check_manipulate(target_scales)
        emb = self.cond_stage_model.encode(reference, "full") if exists(reference) \
            else torch.zeros_like(self.cond_stage_model.encode(control, "full"))

        if manipulate:
            emb = local_manipulate(
                self.cond_stage_model,
                emb,
                targets,
                target_scales,
                anchors,
                controls,
                enhances,
                thresholds_list
            )
        emb = emb[:, 1:]

        crossattn = expand_to_batch_size(bs, emb)
        control = expand_to_batch_size(bs, control)
        control = -control[:, :1]
        uc_crossattn = torch.zeros_like(crossattn)
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
            use_rx: bool = False,
            noise_level: float = 0.,
            manipulation_params = None,
            **kwargs,
    ):
        """
            User interface function.
        """

        self.noise_level = noise_level
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
                s=hook_sketch[:, :1],
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
