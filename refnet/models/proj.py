import torch

from refnet.models.basemodel import CustomizedColorizer, zero_drop
from refnet.modules.proj import GlobalProjection, LocalProjection
from sgm.util import exists



class ColorizeDiffusion(CustomizedColorizer):
    def __init__(
            self,
            unet_config,
            proj_heads,
            rp = 0.,
            use_local = False,
            *args,
            **kwargs
    ):
        context_dim = unet_config.params.context_dim
        super().__init__(unet_config=unet_config, *args, **kwargs)
        self.cls_proj = GlobalProjection(context_dim, proj_heads)
        self.local_proj = LocalProjection(context_dim)

        self.use_local = use_local
        self.rp = rp
        self.model_list.update({
            "cls_proj": self.cls_proj,
            "local_proj": self.local_proj,
        })
        self.switch_modules = ["cls_proj", "local_proj"]

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
            c = self.cond_stage_model.encode(xc, "local" if self.use_local else "cls").detach()
            if self.training and self.rp > 0:
                # This option is for retrieval-oriented training
                c = torch.cat([c, zero_drop(c, self.rp) * torch.roll(c, 1, dims=0)], 1)

        c = self.local_proj(c) if self.use_local else self.cls_proj(c)
        out = [z, dict(c_crossattn=[c], c_concat=[xs])]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs])
        return out

    def get_trainable_params(self):
        if not self.use_local:
            params = list(self.model.parameters()) + list(self.cls_proj.parameters())
        else:
            params = list(self.local_proj.parameters())
        return params


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

        token_key = "full" if use_local else "cls"
        manipulate = self.check_manipulate(target_scales)
        emb = self.cond_stage_model.encode(reference, token_key) if exists(reference) \
            else torch.zeros_like(self.cond_stage_model.encode(control, token_key))

        if manipulate:
            emb = global_manipulate(self.cond_stage_model, emb[:, :1], targets, target_scales, anchors, enhances)

        crossattn = self.local_proj(emb[:, 1:]) if use_local else self.cls_proj(emb)

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