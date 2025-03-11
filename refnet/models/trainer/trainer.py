import torch
import torch.nn.functional as F

from refnet.util import exists, default, append_dims, zero_drop, warp_resize
from refnet.ldm.ema import LitEma
from refnet.models.basemodel import CustomizedColorizer


class BaseTrainer(CustomizedColorizer):
    def __init__(
            self,
            first_stage_key = "image",
            cond_stage_key = "reference",
            ucg_rate = 0.,
            noisy_training = False,
            offset_noise_level = 0.,
            snr_gamma = 0.,
            ucg_range = 0.,
            warp_p = 0.,
            load_only_unet = False,
            make_it_fit = False,
            use_ema = False,
            masked_loss = False,
            masked_weight = 1.,
            mask_threshold = 0.05,
            *args,
            **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(*args, **kwargs)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.ucg_rate = ucg_rate
        self.ucg_range = ucg_range
        self.noisy_training = noisy_training
        self.offset_noise_level = offset_noise_level
        self.snr_gamma = snr_gamma
        self.masked_loss = masked_loss
        self.masked_weight = masked_weight
        self.mask_threshold = mask_threshold
        self.warp_p = warp_p
        self.make_it_fit = make_it_fit
        self.use_ema = use_ema

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        if reset_ema: assert exists(ckpt_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

    def p_losses(self, x_start, cond, noise=None, t=None):
        t = default(t, torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long())
        cond, t = self.timedepend_preprocess(cond, t)

        # loss calculation is refined according to sdxl
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.offset_noise_level > 0.:
            noise += self.offset_noise_level * append_dims(
                torch.randn((x_start.shape[0], x_start.shape[1]), device=x_start.device), x_start.ndim
            )
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise).to(self.dtype)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            target = noise

        if self.snr_gamma > 0:
            snr = append_dims(
                (self.sqrt_alphas_cumprod[t] / self.sqrt_one_minus_alphas_cumprod[t]) ** 2.,
                t.ndim
            )
            if self.parameterization == "v":
                snr += 1

            mse_loss_weights = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(t)], dim=1
                ).min(dim=1)[0]
                / snr
            ).float()
            loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
            loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights)
            loss = loss.mean()

        else:
            if self.masked_loss:
                mask = cond.pop("loss_mask")
                loss = F.mse_loss(model_output.float(), target.float(), reduction="none") * mask
                loss = loss.mean()
            else:
                loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")

        return loss

    def training_step(self, batch):
        x, c = self.get_input(batch)
        loss = self.p_losses(x, c)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def get_trainable_params(self):
        raise NotImplementedError()

    def on_train_start(self):
        self.dtype = self.first_stage_model.dtype
        self.model.diffusion_model.dtype = self.dtype

    def timedepend_preprocess(self, cond, t):
        # t = (1 - (t/(self.num_timesteps-1)) ** 3.) * (self.num_timesteps - 1)
        # t = t.long()
        crossattn = cond["context"][0]
        if self.noisy_training:
            crossattn = self.q_sample(
                x_start = crossattn,
                t = torch.where(t >= self.num_timesteps * 0.9, self.num_timesteps-t-1, t)
            ).to(self.dtype)

        if self.ucg_rate > 0:
            crossattn = zero_drop(
                crossattn,
                # self.ucg_rate + self.ucg_range * t / (self.num_timesteps - 1)
                self.ucg_rate
            ) * crossattn
        cond["context"] = [crossattn]
        return cond, t

    def log_images(self):
        raise NotImplementedError()


class ColorizerTrainer(BaseTrainer):
    def __init__(
            self,
            control_key = "control",
            mask_key = "mask",
            control_drop = 0.0,
            p_white_bg = 0.,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_key = control_key
        self.mask_key = mask_key
        self.control_drop = control_drop
        self.p_white_bg = p_white_bg

    def get_trainable_params(self):
        params = (list(self.model.parameters()) +
                  list(self.control_encoder.parameters()) +
                  list(self.proj.parameters()))
        return params

    def adjust_mask_threshold(self, threshold):
        for layer in self.attn_layers:
            layer.mask_threshold = threshold
        for module in self.attn_modules["modules"]:
            module.mask_threhsold = threshold

    def get_input(
            self,
            batch,
            bs = None,
            return_inputs = False,
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
            xmr, xms = None, None

            x, xc, xs = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                (x, xc, xs)
            )

            if self.p_white_bg > 0:
                xmr = batch["r" + self.mask_key]
                xms = batch["s" + self.mask_key]
                xmr, xms = map(
                    lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype),
                    (xmr, xms)
                )

                thresh = torch.rand([bs], device=xc.device)
                white_bg_idx = append_dims(torch.rand((bs,), device=xc.device) < self.p_white_bg, xc.ndim)
                white_bg = torch.ones_like(xc)
                xc = torch.where(
                    white_bg_idx,
                    torch.where(xmr > append_dims(thresh, xc.ndim), xc, warp_resize(white_bg, xc.shape[2:])),
                    x
                )

                x = torch.where(
                    white_bg_idx,
                    torch.where(xms > append_dims(thresh, x.ndim), x, warp_resize(white_bg, x.shape[2:])),
                    x
                )
            c = self.get_learned_embedding(xc, self.token_type, warp_p=self.warp_p)
            z = self.get_first_stage_encoding(x)

            if self.control_drop > 0:
                xs = torch.where(zero_drop(xs, self.control_drop) > 0, xs, -torch.ones_like(xs))

        c = self.proj(c)
        control = self.control_encoder(xs)

        out = [z, dict(context=[c.to(self.dtype)], control=control)]
        if return_inputs:
            out.extend([x, xc, xs, xms, xmr])
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
            return_inputs=return_inputs,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x, xc, xs, xms, xmr = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = xc
            log["reconstruction"] = self.decode_first_stage(z.to(self.first_stage_model.dtype))

            if exists(xms):
                log["smask"] = (xms - 0.5) / 0.5
            if exists(xmr):
                log["rmask"] = (xmr - 0.5) / 0.5

        else:
            z, c = out

        crossattn = c["context"][0]
        B, _, H, W = z.shape
        uc_full = c.copy()
        if unconditional_guidance_scale > 1.:
            uc_cross = torch.zeros_like(crossattn)
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