import torch

from refnet.models.basemodel import CustomizedWrapper, CustomizedColorizer
from refnet.util import exists, expand_to_batch_size, instantiate_from_config
from refnet.modules.lora import LoraModules
from refnet.sampling.manipulation import global_manipulate, local_manipulate



class InferenceWrapper(CustomizedColorizer, CustomizedWrapper):
    def __init__(
            self,
            lora_config,
            transformer_config,
            thresh_embedder_config,
            *args,
            **kwargs
    ):
        CustomizedColorizer.__init__(self, *args, **kwargs)
        CustomizedWrapper.__init__(self)

        self.thresh_embedder = instantiate_from_config(thresh_embedder_config)
        self.loras = LoraModules(self.model.diffusion_model, **lora_config)
        self.model_list.update({"embedder": self.thresh_embedder})
        self.switch_cond_modules += ["embedder"]

        self.transformer = instantiate_from_config(transformer_config)
        self.model_list.update({"transformer": self.transformer})
        self.switch_cond_modules += ["transformer"]


    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            control_scale = 1,
            mask_scale = 1,
            cond_aug = 0.,
            merge_scale = 0.,
            fg_scale = 1.,
            bg_scale = 1.,
            smask = None,
            rmask = None,
            mask_threshold_ref = 0.,
            mask_threshold_sketch = 0.,
            background = None,
            fg_enhance = False,
            bg_enhance = False,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            *args,
            **kwargs
    ):
        # fig2fig: Using figure-only image for figure colorization
        # figws2fig: Using figure image with backgrounds for figure colorization

        manipulate = self.check_manipulate(target_scales)
        c = {}
        uc = [{}, {}]

        if exists(reference):
            reference = reference + cond_aug * torch.randn_like(reference)
            emb = self.cond_stage_model.encode(reference, "full")
        else:
            emb = torch.zeros_like(self.cond_stage_model.encode(sketch, "full"))

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

        if bg_enhance:
            assert exists(rmask) and exists(smask)
            threshold_embedding, threshold_embedding_sketch = self.thresh_embedder(
                torch.Tensor([mask_threshold_ref, mask_threshold_sketch]).cuda()
            ).to(emb.device).chunk(2)
            self.loras.switch_lora(False, True)

            # reference mask to split character and background
            if exists(background):
                bg_emb = self.get_learned_embedding(background, "local") + threshold_embedding
            else:
                bg_emb = self.cond_stage_model.encode(
                    torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference)), "local"
                ) + threshold_embedding

                bg_emb = self.transformer(bg_emb)
            emb = torch.cat([emb + threshold_embedding_sketch, bg_emb], 1)
            # sketch mask for cross-attention
            smask = expand_to_batch_size(smask.to(self.dtype), bs)
            for d in [c] + uc:
                d.update({"mask": smask})

        if fg_enhance:
            self.loras.switch_lora(True, False)
            smask = expand_to_batch_size(smask.to(self.dtype), bs)
            emb = emb.repeat(1, 2, 1)
            for d in [c] + uc:
                d.update({"mask": smask})

        if bg_enhance or fg_enhance:
            self.adjust_fgbg_scale(fg_scale, bg_scale, merge_scale, mask_threshold_sketch)
        if fg_enhance and bg_enhance:
            self.loras.switch_lora(True, True)

        else:
            self.loras.deactivate_lora_weights()

        sketch = sketch.to(self.dtype)
        crossattn = expand_to_batch_size(emb, bs).to(self.dtype)
        ucontext = torch.zeros_like(crossattn)

        encoded_sketch = self.control_encoder(
            torch.cat([sketch, -torch.ones_like(sketch)], 0)
        )

        control = []
        uc_control = []
        for es in encoded_sketch:
            es = es * control_scale
            ec, uec = es.chunk(2)
            control.append(expand_to_batch_size(ec, bs))
            uc_control.append(expand_to_batch_size(uec, bs))

        c.update({"control": control, "context": [crossattn]})
        uc[0].update({"control": control, "context": [ucontext]})
        uc[1].update({"control": uc_control, "context": [crossattn]})
        return c, uc