import torch

from refnet.models.basemodel import CustomizedColorizer, zero_drop
from refnet.modules.proj import GlobalProjection
from sgm.util import exists, append_dims, default



class ColorizeDiffusion(CustomizedColorizer):
    def __init__(
            self,
            unet_config,
            proj_heads,
            rp = 0.,
            *args,
            **kwargs
    ):
        context_dim = unet_config.params.context_dim
        super().__init__(unet_config=unet_config, *args, **kwargs)
        self.cls_proj = GlobalProjection(context_dim, proj_heads)

        self.rp = rp
        self.model_list.update({"cls_proj": self.cls_proj})
        self.switch_modules = ["cls_proj"]

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
            z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            c = self.cond_stage_model.encode(xc, "full").detach()

        c = torch.cat([c[:, 1:], self.cls_proj(c[:, :1])], 1)
        out = [z, dict(c_crossattn=[c], c_concat=[xs])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs])
        return out

    def get_trainable_params(self):
        params = list(self.model.parameters()) + list(self.cls_proj.parameters())
        return params

    def p_losses(self, x_start, cond, t, noise=None):
        # add noise to reference embedding & drop reference condition here.
        crossattn = cond["c_crossattn"][0]
        if self.noisy_training:
            local = crossattn[:, :256]
            local = self.q_sample(x_start=local, t=t).to(self.dtype)
            local += self.emb_offset_noise_level * append_dims(
                torch.randn(local.shape[0], device=local.device), local.ndim
            )
            crossattn[:, :256] = local

        crossattn = zero_drop(crossattn, self.ucg_rate) * crossattn
        cond["c_crossattn"]  = [crossattn]

        # loss calculation is refined  according to sdxl
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


from refnet.sampling.manipulation import global_manipulate
class InferenceWrapper(ColorizeDiffusion):
    def prepare_conditions(
            self,
            bs,
            control,
            reference,
            use_local,
            targets,
            anchors,
            target_scales,
            enhances,
            **kwargs
    ):
        def expand_to_batch_size(bs, x):
            x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
            return x

        # manipulate = self.check_manipulate(target_scales)
        emb = self.cond_stage_model.encode(reference, "full") if exists(reference) \
            else torch.zeros_like(self.cond_stage_model.encode(control, "full"))

        # if manipulate:
        #     emb = global_manipulate(self.cond_stage_model, emb[:, :1], targets, target_scales, anchors, enhances)

        crossattn = torch.cat([emb[:, 1:], self.cls_proj(emb[:, :1])], 1)
        crossattn = expand_to_batch_size(bs, crossattn)
        control = expand_to_batch_size(bs, control)

        uc_crossattn = self.get_unconditional_conditioning(crossattn)
        null_control = self.get_unconditional_conditioning(control)
        c = {"c_concat": [control], "c_crossattn": [crossattn]}
        uc = [
            {"c_concat": [control], "c_crossattn": [uc_crossattn]},
            {"c_concat": [null_control], "c_crossattn": [crossattn]}
        ]
        return c, uc