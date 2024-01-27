from .trainer import *
from refnet.modules.lora import LoraModules
from refnet.util import instantiate_from_config


class DualMaskTrainer(ColorizerTrainer):
    def __init__(
            self,
            transformer_config,
            thresh_embedder_config,
            lora_config = None,
            mask_key = "mask",
            thresh_max = 0.95,
            thresh_min = 0.05,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self.thresh_min = thresh_min
        self.thresh_rate = thresh_max - thresh_min
        self.transformer = instantiate_from_config(transformer_config)
        self.thresh_embedder = instantiate_from_config(thresh_embedder_config)

        if exists(lora_config):
            self.loras = LoraModules(self.model.diffusion_model, **lora_config)
        self.control_encoder.eval()
        for p in self.control_encoder.parameters():
            p.requires_grad = False

    def adjust_mask_threshold(self, threshold):
        for layer in self.attn_layers:
            layer.mask_threshold = threshold


    def get_input(
            self,
            batch,
            bs = None,
            return_x = False,
            return_original_cond = False,
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
            xmr = batch['r' + self.mask_key]
            xms = batch['s' + self.mask_key]

            # Convert inputs dtype
            x, xc, xs, xmr, xms = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs, xmr, xms)
            )

            # Prepare foreground and background embeddings
            thresh_s, thresh_r = (
                    self.thresh_min + self.thresh_rate * torch.rand((xmr.shape[0] * 2), device=x.device)
            ).chunk(2)
            c = torch.cat([
                xc,
                torch.where(xmr < append_dims(thresh_r, xmr.ndim), xc, torch.ones_like(xc))
            ], 0)
            c, c_bg = torch.chunk(
                self.cond_stage_model.encode(c, self.token_type, warp_p=self.warp_p).detach(), 2
            )

            # Encode sketch features
            if self.control_drop > 0:
                drop = zero_drop(xs, self.control_drop)
                xs = torch.where(drop > 0, xs, -torch.ones_like(xs))
            control = self.control_encoder(xs)

            # Get ground truth latents
            z = self.get_first_stage_encoding(x).detach()

        # Projecting background embeddings for masked regions
        thresh_emb_r = self.thresh_embedder(thresh_r, self.dtype)
        c_bg = self.transformer(c_bg + thresh_emb_r)

        # Activate loras
        self.adjust_mask_threshold(thresh_s)
        c = torch.cat([c, c_bg], 1)
        out = [z, dict(
            context = [c.to(self.dtype)],
            control = control,
            mask = F.interpolate(xms, scale_factor=0.125, mode="bicubic")
        )]
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.extend([xc, xs, xmr, xms])
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
            bs=N,
            return_x=return_inputs,
            return_original_cond=return_inputs,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x, xc, xs, xmr, xms = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = xc
            log["rmask"] = (xmr - 0.5) / 0.5
            log["smask"] = (xms - 0.5) / 0.5
            log["reconstruction"] = self.decode_first_stage(z.to(self.dtype))
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
        x_samples = self.decode_first_stage(samples.to(self.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log

    def get_trainable_params(self):
        params = (self.loras.get_trainable_lora_weights() +
                  list(self.thresh_embedder.parameters()) +
                  list(self.transformer.parameters()))
        return params