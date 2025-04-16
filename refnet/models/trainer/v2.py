import numpy.random as random
import torch
import torch.nn.functional as F

from .trainer import ColorizerTrainer
from refnet.util import instantiate_from_config, append_dims, exists, warp_resize, background_bleaching
from refnet.modules.reference_net import hack_unet_forward, hack_inference_forward
from refnet.modules.lora import LoraModules

class Trainer(ColorizerTrainer):
    def __init__(
            self,
            training_stage,
            bg_encoder_config,
            style_encoder_config,
            ratio_embedder_config,
            mask_key = "mask",
            thresh_max = 0.99,
            thresh_min = 0.01,
            lora_config = None,
            p_white_bg = 0.,
            p_bg_enhnace = 0.,
            merge_offset = 2,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert training_stage in [0, 1, 2], \
            ("There are 3 training stages in total. "
             "0: Train background encoder and LoRAs"
             "1: Train style encoder"
             "2: Validation")

        self.mask_key = mask_key
        self.thresh_min = thresh_min
        self.thresh_rate = thresh_max - thresh_min
        self.p_white_bg = p_white_bg
        self.p_bg_enhance = p_bg_enhnace if training_stage > 0 else 0
        self.merge_offset = merge_offset
        self.input_size = self.image_size * 8
        self.training_stage = training_stage

        self.loras = LoraModules(self.model.diffusion_model.output_blocks, **lora_config)
        self.bg_encoder = instantiate_from_config(bg_encoder_config)
        self.ratio_embedder = instantiate_from_config(ratio_embedder_config)

        if training_stage >= 1:
            self.style_encoder = instantiate_from_config(style_encoder_config)
            self.bg_encoder.eval()
            for p in list(self.bg_encoder.parameters()) + list(self.loras.get_trainable_lora_weights()):
                p.requires_grad = False

        if training_stage == 2:
            # validation mode
            self.model.diffusion_model.eval()
            self.style_encoder.eval()
            hack_inference_forward(self.style_encoder)
            for p in list(self.model.diffusion_model.parameters()) + list(self.style_encoder.parameters()):
                p.requires_grad = False

        else:
            for p in (list(self.model.diffusion_model.input_blocks.parameters()) +
                      list(self.model.diffusion_model.middle_block.parameters())):
                p.requires_grad = False

        hack_unet_forward(self.model.diffusion_model)
        self.control_encoder.eval()
        for p in self.control_encoder.parameters():
            p.requires_grad = False

    def on_train_start(self):
        super().on_train_start()
        self.bg_encoder.dtype = self.dtype
        if self.training_stage >= 1:
            self.style_encoder.dtype = self.dtype

    def get_trainable_params(self):
        if self.training_stage == 0:
            params = (list(self.loras.get_trainable_lora_weights()) +
                      list(self.bg_encoder.parameters()) +
                      list(self.model.diffusion_model.warp_modules.parameters()))
        else:
            params = (list(self.style_encoder.parameters()) +
                      list(self.model.diffusion_model.style_modules.parameters()))

        return params

    def adjust_mask_threshold(self, threshold):
        for layer in self.attn_layers:
            layer.mask_threshold = threshold

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

    def get_input(
            self,
            batch,
            bs = None,
            return_x = False,
            return_original_cond = False,
            *args,
            **kwargs
    ):
        with (torch.no_grad()):
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]
            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key]
            xms, xmr, xc_bg = None, None, None

            ratio = batch["size"] / self.input_size
            rh, rw = ratio[:, 0], ratio[:, 1]

            # Convert inputs dtype
            x, xc, xs = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs)
            )

            c = self.get_learned_embedding(xc, self.token_type)
            hybrid_emb = torch.cat(self.ratio_embedder(torch.cat([rh, rw], 0)).chunk(2), 1)
            cond = dict({"context": [c]})

        # TODO: background and style training parts will be improved in XL version.
        if self.training_stage == 0 or random.rand() < self.p_bg_enhance:
            xmr = batch['r' + self.mask_key]
            xms = batch['s' + self.mask_key]
            x, xc, xs = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs)
            )

            bs = xms.shape[0]
            # Prepare foreground and background embeddings
            thresh_s, thresh_r = (
                    self.thresh_min + self.thresh_rate * torch.rand((bs * 2), device=x.device)
            ).chunk(2)

            # Mask merging
            xmr = torch.roll(xmr, self.merge_offset, 0) + xmr
            xmr = xmr.clamp(0, 1)

            # Background bleaching
            x, xs, xc_bg = background_bleaching(x, xs, xc, xms, xmr, thresh_s, thresh_r, self.p_white_bg, self.dtype)
            z_bg = self.get_first_stage_encoding(warp_resize(xc_bg, (x.shape[2], x.shape[3])))
            c_bg = self.get_learned_embedding(xc_bg, self.token_type)
            c_concat = torch.cat([c, c_bg], 1)

            self.adjust_mask_threshold(append_dims(thresh_s, c.ndim))
            bgs = self.bg_encoder(
                x = z_bg,
                timesteps = torch.zeros((bs,), dtype=torch.long, device=z_bg.device),
                y = hybrid_emb,
                # TODO: Unnecessary. Will be removed in XL version.
                context = c,
            )

            cond.update({
                "context": [c_concat],
                "mask": F.interpolate(xms, scale_factor=0.125, mode="bicubic"),
                "threshold": append_dims(thresh_s, z_bg.ndim),
                "hs_bg": bgs
            })

        if self.training_stage > 0:
            # style enhance training
            zc = self.get_first_stage_encoding(xc)
            modulations = self.style_encoder(
                x = zc,
                timesteps = torch.zeros((bs,), dtype=torch.long, device=zc.device),
                y = hybrid_emb.to(self.dtype),
                # TODO: Unnecessary. Will be removed in XL version.
                context = c,
            )
            cond.update({"style_modulations": modulations})

        control = self.control_encoder(xs)
        cond.update({"control": control})
        z = self.get_first_stage_encoding(x)
        out = [z, cond]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xc_bg, xs, xmr, xms])
        return out

    @torch.no_grad()
    def log_images(
            self,
            batch,
            N = 4,
            sampler = "dpm_vp",
            step = 20,
            unconditional_guidance_scale = 9.0,
            return_inputs = False,
            **kwargs
    ):
        """
            This function is used for batch processing.
            Used with image logger.
        """

        out = self.get_input(
            batch,
            bs = N,
            return_x = return_inputs,
            return_original_cond = return_inputs,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x, xc, xc_bg, xs, xmr, xms = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.first_stage_model.dtype))

            if exists(xmr):
                log["rmask"] = (xmr - 0.5) / 0.5
            if exists(xms):
                log["smask"] = (xms - 0.5) / 0.5
            if exists(xc_bg):
                log["bg_conditioning"] = xc_bg
        else:
            z, c = out

        crossattn = c["context"][0]
        B, _, H, W = z.shape

        if unconditional_guidance_scale > 1.:
            uc_cross = torch.zeros_like(crossattn)
            uc_full = c
            uc_full.update({"context": [uc_cross]})
        else:
            uc_full = None

        samples = self.sample(
            cond = c,
            bs = B,
            shape = (self.channels, H, W),
            step = step,
            sampler = sampler,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            device = z.device,
        )
        x_samples = self.decode_first_stage(samples.to(self.first_stage_model.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log
