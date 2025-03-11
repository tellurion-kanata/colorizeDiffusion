import torch

from refnet.models.basemodel import CustomizedColorizer, CustomizedWrapper, torch_dfs
from refnet.util import exists, expand_to_batch_size
from refnet.sampling.manipulation import local_manipulate, global_manipulate


class InferenceWrapper(CustomizedWrapper, CustomizedColorizer):
    def __init__(self, *args, **kwargs):
        CustomizedColorizer.__init__(self, *args, **kwargs)
        CustomizedWrapper.__init__(self)


    def adjust_masked_attn(self, mask_threshold):
        for layer in self.attn_layers:
            layer.mask_threshold = mask_threshold


    def get_learned_embedding(self, *args, **kwargs):
        return super().get_learned_embedding(*args, **kwargs).to(self.dtype)


    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            control_scale = 1,
            mask_scale = 1,
            cond_aug = 0.,
            smask = None,
            rmask = None,
            mask_threshold_ref = 0.,
            mask_threshold_sketch = 0.,
            style_enhance = False,
            bg_enhance = False,
            fg_scale = 1.,
            bg_scale = 1.,
            merge_scale = 0.,
            background = None,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            *args,
            **kwargs
    ):
        # prepare reference embedding
        manipulate = self.check_manipulate(target_scales)
        c = {}
        uc = [{}, {}]
        
        if exists(reference):
            reference = reference + cond_aug * torch.randn_like(reference)
            emb = self.cond_stage_model.encode(reference, "full")
        else:
            emb = torch.zeros_like(self.cond_stage_model.encode(sketch, "full"))

        # text manipulation
        if self.token_type == "cls":
            if manipulate:
                emb = global_manipulate(
                    self.cond_stage_model,
                    emb[:, :1],
                    targets,
                    target_scales,
                    anchors,
                    enhances,
                )
            emb = emb[:, :1]
        else:
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

        emb = self.proj(emb)
        # masked attention preprocessing
        if bg_enhance:
            assert exists(rmask) and exists(smask)

            self.adjust_fgbg_scale(fg_scale, bg_scale, merge_scale, mask_threshold_sketch)
            if exists(background):
                bg_emb = self.get_learned_embedding(background, "local")
            else:
                bg_emb = self.get_learned_embedding(
                    torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference)), "local"
                )
            emb = torch.cat([emb, bg_emb], 1)

            # sketch mask for cross-attention
            smask = expand_to_batch_size(smask.to(self.dtype), bs)

            for d in [c] + uc:
                d.update({"mask": smask})

        sketch = sketch.to(self.dtype)
        context = expand_to_batch_size(emb, bs).to(self.dtype)
        uc_context = torch.zeros_like(context)

        control = []
        uc_control = []
        encoded_sketch = self.control_encoder(
            torch.cat([sketch, -torch.ones_like(sketch)], 0)
        )
        for es in encoded_sketch:
            es = es * control_scale
            ec, uec = es.chunk(2)
            control.append(expand_to_batch_size(ec, bs))
            uc_control.append(expand_to_batch_size(uec, bs))

        c.update({"control": control, "context": [context]})
        uc[0].update({"control": control, "context": [uc_context]})
        uc[1].update({"control": uc_control, "context": [context]})
        return c, uc