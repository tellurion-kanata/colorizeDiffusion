import torch

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import CondTransformer
from utils import instantiate_from_config

class PluginConditionWrapper(LatentDiffusion):
    def __init__(self, control_stage_config, only_mid_control, input_dim, context_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_mapper = CondTransformer(input_dim, context_dim)
        self.control_model = instantiate_from_config(control_stage_config)
        self.only_mid_control = only_mid_control

    def get_learned_conditioning(self, c):
        c = super().get_learned_conditioning(c)

        context = c["c_crossattn"][0]
        context = self.cond_mapper(context)
        c["c_crossattn"] = [context]
        return c

    def get_unconditional_conditioning(self, c, label="reference"):
        assert label in ["sketch", "reference"]
        if label == "reference":
            if self.use_adm:
                c_adm = c["c_adm"][0]
                uc = {"c_concat": c["c_concat"], "c_adm": [torch.zeros_like(c_adm, device=c_adm.device)]}
            else:
                crossattn = c["c_crossattn"][0]
                uc = {"c_concat": c["c_concat"], "c_crossattn": [torch.zeros_like(crossattn, device=crossattn.device)]}
        else:
            raise NotImplementedError
        return uc

    def configure_optimizers(self):
        lr = self.lr
        opt = torch.optim.AdamW(self.cond_mapper.parameters(), lr=lr)
        return opt

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps