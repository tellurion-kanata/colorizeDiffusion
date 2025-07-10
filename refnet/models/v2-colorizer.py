import torch

from refnet.models.basemodel import CustomizedWrapper, CustomizedColorizer
from refnet.util import (
    instantiate_from_config, exists, expand_to_batch_size, warp_resize, resize_and_crop, default,
    get_crop_scale
)
from refnet.modules.reference_net import hack_unet_forward, hack_inference_forward
from refnet.modules.lora import LoraModules
from refnet.sampling.manipulation import global_manipulate, local_manipulate


class InferenceWrapper(CustomizedColorizer, CustomizedWrapper):
    def __init__(
            self,
            bg_encoder_config,
            style_encoder_config,
            ratio_embedder_config,
            lora_config = None,
            *args,
            **kwargs
    ):
        CustomizedColorizer.__init__(self, *args, **kwargs)
        CustomizedWrapper.__init__(self)

        self.loras = LoraModules(self.model.diffusion_model.output_blocks, **lora_config)
        self.bg_encoder = instantiate_from_config(bg_encoder_config)
        self.style_encoder = instantiate_from_config(style_encoder_config)
        self.ratio_embedder = instantiate_from_config(ratio_embedder_config)
        new_model_list = {
            "rembedder": self.ratio_embedder,
            "bg_encoder": self.bg_encoder,
            "style_encoder": self.style_encoder,
        }
        self.switch_cond_modules += list(new_model_list.keys())
        self.model_list.update(new_model_list)

        hack_unet_forward(self.model.diffusion_model)
        hack_inference_forward(self.bg_encoder)
        hack_inference_forward(self.style_encoder)

    def switch_to_fp16(self):
        super().switch_to_fp16()
        self.model.diffusion_model.warp_modules.to(self.half_precision_dtype)
        self.model.diffusion_model.style_modules.to(self.half_precision_dtype)
        self.style_encoder.to(self.half_precision_dtype)
        self.bg_encoder.to(self.half_precision_dtype)
        self.style_encoder.dtype = self.half_precision_dtype
        self.bg_encoder.dtype = self.half_precision_dtype

        self.style_encoder.time_embed.float()
        self.bg_encoder.time_embed.float()

    def switch_to_fp32(self):
        super().switch_to_fp32()
        self.model.diffusion_model.warp_modules.float()
        self.model.diffusion_model.style_modules.float()
        self.style_encoder.float()
        self.bg_encoder.float()
        self.style_encoder.dtype = torch.float32
        self.bg_encoder.dtype = torch.float32


    def get_size_embedding(self, x: torch.Tensor, height, width, device=torch.device("cuda")):
        oh, ow = x.shape[2:]
        if oh < height or ow < width:
            # A simple bias to avoid deterioration caused by reference resolution
            mind = max(height, width)
            ih = oh + mind
            iw = ow / oh * ih
        else:
            ih, iw = oh, ow
        rh, rw = ih / height, iw / width
        hybrid_emb = torch.cat(self.ratio_embedder(
            torch.Tensor([rh, rw]).cuda()
        ).to(device).chunk(2), 1)
        return hybrid_emb.to(self.dtype)

    def get_learned_embedding(self, *args, **kwargs):
        return super().get_learned_embedding(*args, **kwargs).to(self.dtype)

    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            height,
            width,
            control_scale,
            mask_scale = 1,
            merge_scale = 1.,
            cond_aug = 0.,
            background = None,
            smask = None,
            rmask = None,
            mask_threshold_ref = 0.,
            mask_threshold_sketch = 0.,
            style_enhance = False,
            fg_enhance = False,
            bg_enhance = False,
            latent_inpaint = False,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            *args,
            **kwargs
    ):
        def prepare_style_modulations():
            z_ref = self.get_first_stage_encoding(warp_resize(reference, (height, width)))

            # Style enhancement part
            if exists(background):
                z_bg = self.get_first_stage_encoding(warp_resize(background, (height, width)))
                bg_emb = self.get_learned_embedding(background, "local")
                size_emb = self.get_size_embedding(background, height, width, z_ref.device)
                style_modulations = self.style_encoder(
                    torch.cat([z_ref, z_bg]),
                    timesteps = torch.zeros((2,), dtype=torch.long, device=z_ref.device),
                    context = torch.cat([emb, bg_emb]),
                    y = torch.cat([
                        self.get_size_embedding(reference, height, width, z_ref.device),
                        size_emb
                    ])
                )

                for idx, m in enumerate(style_modulations):
                    fg, bg = m.chunk(2)
                    m = fg * (1-merge_scale) + merge_scale * bg
                    style_modulations[idx] = expand_to_batch_size(m, bs).to(self.dtype)

            else:
                z_bg = None
                bg_emb = None
                size_emb = self.get_size_embedding(reference, height, width, emb.device)
                style_modulations = self.style_encoder(
                    z_ref,
                    timesteps = torch.zeros((1,), dtype=torch.long, device=z_ref.device),
                    context = emb,
                    y = size_emb,
                )
                style_modulations = [expand_to_batch_size(m, bs).to(self.dtype) for m in style_modulations]

            return style_modulations, z_bg, bg_emb, size_emb

        def prepare_background_latents(z, bg_emb, size_emb):
            if latent_inpaint and exists(background):
                bgh, bgw = background.shape[2:] if exists(background) else reference.shape[2:]
                ch, cw = get_crop_scale(height, width, bgh, bgw)
                hs_bg = self.get_first_stage_encoding(resize_and_crop(background, ch, cw, height, width))
                bg_emb = self.get_learned_embedding(background, "local")

            else:
                if not exists(z):
                    if exists(background):
                        z = self.get_first_stage_encoding(warp_resize(background, (height, width)))
                        bg_emb = self.get_learned_embedding(background, "local")
                        size_emb = self.get_size_embedding(background, height, width, bg_emb.device)
                        input_emb = bg_emb

                    else:
                        z = self.get_first_stage_encoding(warp_resize(
                                torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference)),
                                (height, width)
                        ))
                        bg_emb = self.get_learned_embedding(
                            torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference)), "local"
                        )
                        size_emb = self.get_size_embedding(reference, height, width, bg_emb.device)
                        input_emb = emb
                else:
                    input_emb = emb

                hs_bg = self.bg_encoder(
                    x = z,
                    timesteps = torch.zeros((1,), dtype=torch.long, device=z.device),
                    y = size_emb,
                    context = input_emb,
                )
                hs_bg = [hs * mask_scale for hs in hs_bg]
                hs_bg = expand_to_batch_size(hs_bg, bs)
            return hs_bg, bg_emb

        manipulate = self.check_manipulate(target_scales)
        c = {}
        uc = [{}, {}]

        if exists(reference):
            reference = reference + cond_aug * torch.randn_like(reference)
            emb = self.get_learned_embedding(reference, "full")

        else:
            emb = torch.zeros_like(self.get_learned_embedding(sketch, "full"))

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

        z_bg, size_emb, bg_emb = None, None, None
        self.loras.switch_lora(False, False)
        if style_enhance:
            style_modulations, z_bg, bg_emb, size_emb = prepare_style_modulations()
            for d in [c] + uc:
                d.update({"style_modulations": style_modulations})

        # Background enhancement part
        if bg_enhance:
            assert exists(smask) and (exists(rmask) or exists(background))
            # Hack U-Net forward to inject latent features
            self.loras.switch_lora(True, True)
            hs_bg, bg_emb = prepare_background_latents(z_bg, bg_emb, size_emb)

            if latent_inpaint:
                c.update({"inpaint_bg": hs_bg})
            else:
                for d in [c] + uc:
                    d.update({"hs_bg": hs_bg})

        if fg_enhance or bg_enhance:
            # need to activate mask-guided split cross-attetnion
            emb = torch.cat([emb, default(bg_emb, emb)], 1)
            smask = expand_to_batch_size(smask.to(self.dtype), bs)
            for d in [c] + uc:
                d.update({"mask": smask, "threshold": mask_threshold_sketch})

        sketch = sketch.to(self.dtype)
        context = expand_to_batch_size(emb, bs).to(self.dtype)
        uc_context = torch.zeros_like(context)

        encoded_sketch = self.control_encoder(
            torch.cat([sketch, -torch.ones_like(sketch)], 0)
        )

        control = []
        uc_control = []
        for idx, es in enumerate(encoded_sketch):
            es = es * control_scale
            ec, uec = es.chunk(2)
            control.append(expand_to_batch_size(ec, bs))
            uc_control.append(expand_to_batch_size(uec, bs))

        c.update({"control": control, "context": [context]})
        uc[0].update({"control": control, "context": [uc_context]})
        uc[1].update({"control": uc_control, "context": [context]})
        return c, uc