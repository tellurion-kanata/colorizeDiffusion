import torch

from refnet.models.basemodel import CustomizedColorizer, CustomizedWrapper
from refnet.util import exists, expand_to_batch_size


from refnet.sampling.manipulation import local_manipulate, global_manipulate
class InferenceWrapper(CustomizedWrapper, CustomizedColorizer):
    def __init__(self, *args, **kwargs):
        CustomizedColorizer.__init__(self, *args, **kwargs)
        CustomizedWrapper.__init__(self)


    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            *args,
            **kwargs
    ):
        manipulate = self.check_manipulate(target_scales)
        emb = self.cond_stage_model.encode(reference, "full") if exists(reference) \
            else torch.zeros_like(self.cond_stage_model.encode(sketch, "full"))

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

        sketch = sketch.to(self.dtype)
        crossattn = expand_to_batch_size(emb, bs).to(self.dtype)
        uc_crossattn = torch.zeros_like(crossattn)

        encoded_sketch = self.control_encoder(
            torch.cat([sketch, -torch.ones_like(sketch)], 0)
        )
        control = []
        uc_control = []
        for c in encoded_sketch:
            c, uc = c.chunk(2)
            control.append(expand_to_batch_size(c, bs))
            uc_control.append(expand_to_batch_size(uc, bs))

        c = {"c_concat": control, "c_crossattn": [crossattn]}
        uc = [
            {"c_concat": control, "c_crossattn": [uc_crossattn]},
            {"c_concat": uc_control, "c_crossattn": [crossattn]}
        ]
        return c, uc