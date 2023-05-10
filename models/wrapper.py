import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import instantiate_from_config
from ldm.modules.encoders.modules import OpenCLIPEncoder, OpenCLIP
from ldm.models.diffusion.ddpm import LatentDiffusion
from models.loss import MappingLoss

def exists(v):
    return v is not None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def sum(v: torch.Tensor):
    return v.sum(dim=1, keepdim=True)

def gap(x: torch.Tensor = None, keepdim=True):
    if len(x.shape) == 4:
        return torch.mean(x, dim=[2, 3], keepdim=keepdim)
    elif len(x.shape) == 3:
        return torch.mean(x, dim=[1], keepdim=keepdim)
    else:
        raise NotImplementedError('gap input should be 3d or 4d tensors')

def maxmin(s: torch.Tensor, threshold=0.5):
    """
        The shape of input scales tensor should be (b, n, 1)
    """
    assert len(s.shape) == 3
    ms = gap(s)
    maxm = s.max(dim=1, keepdim=True).values
    minm = s.min(dim=1, keepdim=True).values
    d = maxm - minm

    corr_s = (s - minm) / d
    corr_mean = (ms - minm) / d
    return torch.where(corr_s - corr_mean > 0, torch.exp(torch.abs(s-ms) * 0.5), -torch.exp(torch.abs(s-ms)))


class ReferenceWrapper(nn.Module):
    def __init__(self,
                 clip_config,
                 pool_config=None,
                 drop_rate=0.,
                 ):
        super().__init__()
        self.encoder = OpenCLIPEncoder(**clip_config)
        self.latent_pooling = instantiate_from_config(pool_config) if exists(pool_config) else None
        self.drop_rate = drop_rate
        self.sample = False

    def encode(self, c):
        """
            return the visual features of reference image,
            shuffle and noise the latent codes during training if using tokens
        """
        z = self.encoder.encode(c).to(c.dtype)
        if exists(self.latent_pooling):
            z = self.latent_pooling(z, self.sample)
        if self.training and self.drop_rate:
            z = torch.bernoulli((1 - self.drop_rate) * torch.ones(z.shape[0], device=z.device)[:, None, None]) * z
        return  {"c_crossattn": [z.detach()]}


class ConditionWrapper(nn.Module):
    OpenCLIPEncoders = {
        "image": OpenCLIPEncoder,
        "full": OpenCLIP,
    }
    def __init__(self,
                 clip_config: dict,
                 n_emb=512,
                 emb_dims=1024,
                 pool_config=None,
                 use_codebook=False,
                 init_dr=0.5,
                 finl_dr=0.5,
                 decl_dr=0.,
                 use_adm=False,
                 encoder_type="image",
                 ):
        super().__init__()
        self.encoder = self.OpenCLIPEncoders[encoder_type](**clip_config)
        self.latent_pooling = instantiate_from_config(pool_config) if exists(pool_config) else None
        self.priorbook = nn.Parameter(torch.randn([n_emb, emb_dims])) if use_codebook else None

        self.use_adm = use_adm
        self.n_emb = n_emb
        self.drop_rate = init_dr
        self.final_drop_rate = finl_dr
        self.drop_rate_decl = decl_dr

    def get_input(self, batch, device):
        x, r = batch["sketch"], batch["reference"]
        x, r = map(lambda t: t.to(memory_format=torch.contiguous_format).to(device), (x, r))
        return {"sketch": x, "reference": r}

    def encode_text(self, text):
        return self.encoder.encode_text(text)

    def calculate_scale(self, v, t):
        return self.encoder.calculate_scale(v, t)

    def update_drop_rate(self):
        if self.drop_rate > self.final_drop_rate:
            self.drop_rate = max(self.drop_rate - self.drop_rate_decl, self.final_drop_rate)

    def forward(self, c):
        """
            wrap conditions
            return the visual features of reference image,
            shuffle and add noise (optionally) to the latent codes during training if using tokens and original color images
        """
        s, r = c["sketch"], c["reference"]
        z = self.encoder.encode(r).to(r.dtype).detach()
        # shuffle and add latent noise to the reference latent codes
        if exists(self.latent_pooling) and self.training:
            z = self.latent_pooling(z)

        # drop reference conditions according to the drop_rate
        if self.training and self.drop_rate:
            z = torch.bernoulli((1 - self.drop_rate) * torch.ones(z.shape[0], device=z.device)[:, None, None]) * z

        c_dict = {"c_concat": [s]}
        # using cross-attention path for visual tokens
        if exists(self.priorbook):
            priorbook = torch.ones([z.shape[0]], device=z.device)[:, None, None] * self.priorbook
            z = torch.cat([z, priorbook], dim=1)
        c_dict.update({"c_crossattn": [z]})
        if self.use_adm:
            c_dict.update({"c_adm": [gap(z, keepdim=False)]})
        return c_dict


from models.mapper import Mapper
class AdjustLatentDiffusion(LatentDiffusion):
    def __init__(self, type="tokens", offset=1, context_dim=1024, learnable=False, *args, **kwargs):
        assert type in ["tokens", "pooled", "full"]
        super().__init__(*args, **kwargs)
        self.type = type
        self.zeroshot = not learnable
        if learnable:
            self.offset = offset
            self.mapper = Mapper(context_dim)
            self.loss = MappingLoss()
            self.model = self.model.eval()
            self.model.train = disabled_train

    def get_input(self, batch, return_first_stage_outputs=False, text=None,
                  return_original_cond=False, bs=None, return_x=False, **kwargs):
        if bs:
            for k in batch:
                batch[k] = batch[k][:bs]

        x = batch["color"]
        ref = batch["reference"]
        s = batch["sketch"]
        idx = batch["index"]
        text = batch["text"] if not exists(text) else [text] * x.shape[0]

        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
        c = self.get_learned_conditioning({"sketch": s, "reference": ref})
        t = self.cond_stage_model.encode_text(text)

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append({"sketch": s, "reference": ref})
        return out, idx, t

    def roll_input(self, v, t):
        """
            Shift the visual features forward to get a set of incorrect image features.
            When adopting tokens as reference visual features, the scale would be (b, n, 1)
        """
        shifted_v = torch.roll(v, self.offset, dims=0)
        shifted_t = torch.roll(t, self.offset, dims=0)
        return shifted_v, shifted_t

    def forward(self, v, t, c, **kwargs):
        """
            v: visual tokens in shape (b, n, c)
            t: target prompts in shape (b, 1, c)
            c: control prompts in shape (b, 1, c)
            s_t: scale for target prompts in shape (b, 1, 1)
        """
        cls_token = v[:, 0].unsqueeze(1)
        c, t = map(lambda x: torch.cat([x, self.cond_stage_model.calculate_scale(cls_token, x)], dim=2), (c, t))
        t = torch.cat([c, t], dim=1)
        return self.mapper(v, t)

    def training_step(self, batch, batch_idx):
        out, idx, t = self.get_input(batch)
        x, c = out
        sketch, v = c["c_concat"][0], c["c_crossattn"][0]

        shifted_v, shifted_t = self.roll_input(v, t)
        fake_v = self(shifted_v, t, shifted_t)
        loss, loss_dict = self.loss(x, sketch, fake_v, shifted_v, t, self)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.mapper.parameters(), lr=self.lr)
        return opt

    def manipulate(self, v, target_scale, target, control=None, threshold=0.5):
        """
            v: visual tokens in shape (b, n, c)
            target: target text embeddings in shape (b, 1 ,c)
            control: control text embeddings in shape (b, 1, c)
        """
        if self.type == "pooled":
            for t, c, s_t in zip(target, control, target_scale):
                # remove control prompts
                if c != "None":
                    c = [c] * v.shape[0]
                    c = self.cond_stage_model.encode_text(c)
                    s_c = self.cond_stage_model.calculate_scale(v, c)
                    v = v - s_c * c

                # adjust target prompts
                t = [t] * v.shape[0]
                t = self.cond_stage_model.encode_text(t)
                cur_target_scale = self.cond_stage_model.calculate_scale(gap(v), t)
                print(f"current target scale: {cur_target_scale}")
                v = v + (s_t - cur_target_scale) * t
        else:
            # zero shot spatial manipulation requires corresponding control prompts
            # assert len(target) == len(control)
            cls_token = v[:, 0].unsqueeze(1)
            for t, c, s_t in zip(target, control, target_scale):
                c = [c] * v.shape[0]
                t = [t] * v.shape[0]
                c = self.cond_stage_model.encode_text(c)
                t = self.cond_stage_model.encode_text(t)

                print(f"current target scale: {gap_cur_target_scale}")

                if not self.zeroshot:
                    c, t = map(lambda x: torch.cat([x, self.cond_stage_model(cls_token, x)], dim=2), (c, t))
                    t = torch.cat([c, t], dim=1)
                    v = self.mapper(v, t)[:, 1:]
                else:
                    s_o = self.cond_stage_model.calculate_scale(cls_token, c)
                    pwm = maxmin(s_o)
                    cur_target_scale = self.cond_stage_model.calculate_scale(cls_token, t)
                    gap_cur_target_scale = gap(cur_target_scale)
                    # pwm = pwm / pwm.sum(dim=1, keepdim=True)
                    dscale = s_t - gap_cur_target_scale
                    v = v + dscale * t
                    v = v[:, 1:]
                    # v = v - c * s_o
                    # v = v + (s_c + s_t - 2 * gap_s_c) * t
        return [v]

    def log_images(self, batch, N=8, control=[], target=[], target_scale=[], threshold=0.5, is_train=False,
                   return_inputs=True, sample_original_cond=True, unconditional_guidance_scale=1.0, **kwargs):
        if len(target) > 0 or is_train:
            if is_train:
                out, idx, t = self.get_input(batch,
                                             return_first_stage_outputs=return_inputs,
                                             return_original_cond=return_inputs,
                                             bs=N)
                z, c = out[:2]
                v = c["c_crossattn"][0]
                shift_v, shift_t = self.roll_input(v, t)
                adjust_v = [self(shift_v, t, shift_t)[:, 1:]]
            else:
                assert len(target) == len(target_scale), "Each prompt should have a target scale"
                out, idx = super().get_input(batch, self.first_stage_key,
                                             return_first_stage_outputs=return_inputs,
                                             cond_key=self.cond_stage_key,
                                             force_c_encode=True,
                                             return_original_cond=return_inputs,
                                             bs=N)
                z, c = out[:2]
                adjust_v = self.manipulate(c["c_crossattn"][0], target_scale, target, control)

            log = {}
            x_T = torch.randn_like(z, device=z.device)
            if sample_original_cond:
                inputs = [out, idx, x_T]
                original_log, _ = super().log_images(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                                     **kwargs)
                original_sample_key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}" \
                    if unconditional_guidance_scale > 1.0 else "samples"
                log.update({"original_sample": original_log[original_sample_key]})

            out[1]["c_crossattn"] = adjust_v
            inputs = [out, idx, x_T]
            sample_log, idx = super().log_images(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                                 unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
            log.update(sample_log)
            return log, idx
        else:
            return super().log_images(batch=batch, N=N, return_inputs=return_inputs,
                                      unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)