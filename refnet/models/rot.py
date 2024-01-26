import torch

from refnet.models.basemodel import CustomizedColorizer
from refnet.modules.encoders import FilterAttention
from sgm.util import exists



class ColorizeDiffusion(CustomizedColorizer):
    def __init__(
            self,
            unet_config,
            *args,
            **kwargs
    ):
        context_dim = unet_config.params.context_dim
        super().__init__(unet_config=unet_config, *args, **kwargs)
        self.filter = FilterAttention(context_dim)
        self.model_list.update({"filter": self.filter})
        self.switch_modules = ["filter"]

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
            c, s = self.cond_stage_model.encode(torch.cat([xc, -xs], 0), "local").detach().chunk(2)
            if self.training:
                c = torch.cat([c, torch.roll(c, 1, dims=0)], dim=1)

        c = self.filter(c, s)
        out = [z, dict(c_crossattn=[c], c_concat=[xs])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs])
        return out

    def get_trainable_params(self):
        params = list(self.model.parameters()) + list(self.filter.parameters())
        return params


class InferenceWrapper(ColorizeDiffusion):
    def get_learned_conditioning(self, c, s, **kwargs):
        c, s = map(lambda t: self.cond_stage_model.preprocess(t), (c, -s))
        c, s = self.cond_stage_model.encode(torch.cat([c, s], 0), "local", False).detach().chunk(2)
        c = self.filter(c, s)
        return c

    def get_adjusted_conditioning(self, emb, target, anchor, scale, enhance):
        if anchor != "" and anchor != "none":
            anchor = self.cond_stage_model.encode_text(anchor)
            if enhance:
                anchor_scale = self.cond_stage_model.calculate_scale(emb, anchor)
                demb = target * scale - anchor * anchor_scale
            else:
                demb = (target - anchor) * scale
        else:
            demb = target * scale
        return emb + demb

    def prepare_conditions(
            self,
            bs,
            control,
            reference,
            use_local,
            target_texts,
            anchor_texts,
            target_scales,
            enhances
    ):
        def expand_to_batch_size(bs, x):
            x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
            return x

        # manipulate = exists(target_scales) and len(target_scales) > 0
        if exists(reference):
            emb = self.get_learned_conditioning(reference, control)
        else:
            emb = self.get_unconditional_conditioning(
                self.get_learned_conditioning(control, control)
            )

        # if manipulate:
        #     target_prompts = self.cond_stage_model.encode_text(target_texts)
        #     for (target, anchor, scale, enhance) in zip(target_prompts, anchor_texts, target_scales, enhances):
        #         target = target.unsqueeze(0)
        #         emb = self.get_adjusted_conditioning(emb, target, anchor, scale, enhance)

        emb = expand_to_batch_size(bs, emb)
        control = expand_to_batch_size(bs, control)

        null_emb = torch.zeros_like(emb)
        null_control = torch.zeros_like(control)
        c = {"c_concat": [control], "c_crossattn": [emb]}
        uc = [
            {"c_concat": [control], "c_crossattn": [null_emb]},
            {"c_concat": [null_control], "c_crossattn": [emb]}
        ]
        return c, uc