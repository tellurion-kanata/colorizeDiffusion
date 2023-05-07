import torch
import torch.nn as nn

from utils import instantiate_from_config
from ldm.modules.encoders.modules import OpenCLIPEncoder, OpenCLIP
from ldm.models.diffusion.ddpm import LatentDiffusion


def exists(v):
    return v is not None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def max(v: torch.Tensor):
    """
        The shape of input visual tokens should be (b, c, n)
    """
    return v.max(dim=1, keepdim=True)

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
    maxm = s.max(dim=1, keepdim=True).values
    minm = s.min(dim=1, keepdim=True).values
    d = maxm - minm

    s = (s - minm) / d
    # return torch.where(s > threshold, 0, 1-s)
    return s

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
                 drop_rate=0.,
                 use_adm=False,
                 encoder_type="image",
                 ):
        super().__init__()
        self.encoder = self.OpenCLIPEncoders[encoder_type](**clip_config)
        self.latent_pooling = instantiate_from_config(pool_config) if exists(pool_config) else None
        self.priorbook = nn.Parameter(torch.randn([n_emb, emb_dims])) if use_codebook else None

        self.use_adm = use_adm
        self.n_emb = n_emb
        self.drop_rate = drop_rate

    def get_input(self, batch, device):
        x, r = batch["sketch"], batch["reference"]
        x, r = map(lambda t: t.to(memory_format=torch.contiguous_format).to(device), (x, r))
        return {"sketch": x, "reference": r}

    def encode_text(self, text):
        return self.encoder.encode_text(text)

    def calculate_scale(self, v, t):
        return self.encoder.calculate_scale(v, t)
    
    def forward(self, c):
        """
            wrap conditions
            return the visual features of reference image,
            shuffle and add noise (optionally) to the latent codes during training if using tokens and original color images
        """
        s, r = c["sketch"], c["reference"]
        z = self.encoder.encode(r).to(r.dtype).detach()
        # shuffle and add latent noise to the reference latent codes
        if exists(self.latent_pooling):
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


class AdjustLatentDiffusion(LatentDiffusion):
    def __init__(self, type="tokens", *args, **kwargs):
        assert type in ["tokens", "pooled"]
        super().__init__(*args, **kwargs)
        self.type = type

    def adjust_visual(self, v, t, c, s_t, s_c):
        """
            v: visual tokens in shape (b, n, c)
            t: target text embeddings in shape (b, 1 ,c)
            c: control text embeddings in shape (b, 1, c)
            s: position weight matrix in shape (b, n, 1)
        """
        return v + t * s_t - c * s_c

    def manipulation(self, v, target_scale, target, control=None, threshold=0.5):
        if exists(control):
            control = [control] * v.shape[0]
            control = self.cond_stage_model.encode_text(control)
            control_scale = self.cond_stage_model.calculate_scale(v, control)
        else:
            control, control_scale = 0, 1
        target = self.cond_stage_model.encode_text(target)
        cur_target_scale = self.cond_stage_model.calculate_scale(gap(v), target)

        if self.type == "pooled":
            """
                Used for the pooled model which adopts the globally-averaged token.
            """
            dscale = target_scale - cur_target_scale
            print(f"current target scale: {cur_target_scale}")
            v = self.adjust_visual(v, target, control, dscale, control_scale)
        else:
            """
                Used for the token model, which adopts full tokens and requires spatial information.
            """
            dscale = target_scale * control_scale - cur_target_scale
            # tscale = tscale.gather(1, max)
            print(f"current target scale: {gap(cur_target_scale)}")
            v = self.adjust_visual(v, target, control, dscale, control_scale)
        return [v]

    def log_images(self, batch, target_scale=None, N=8, control=None, target=None, threshold=0.5,
                   return_inputs=True, sample_original_cond=True, unconditional_guidance_scale=1.0, **kwargs):
        if exists(target) and exists(target_scale):
            out, idx = self.get_input(batch, self.first_stage_key,
                                      return_first_stage_outputs=return_inputs,
                                      cond_key=self.cond_stage_key,
                                      force_c_encode=True,
                                      return_original_cond=return_inputs,
                                      bs=N)
            z, c = out[:2]
            target = [target] * z.shape[0]
            adjust_crossattn = self.manipulation(c["c_crossattn"][0], target_scale, target, control, threshold)

            log = {}
            x_T = torch.randn_like(z, device=z.device)
            if sample_original_cond:
                inputs = [out, idx, x_T]
                original_log, _ = super().log_images(batch=None, inputs=inputs, N=N, return_inputs=False,
                                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                                     **kwargs)
                original_sample_key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}" \
                    if unconditional_guidance_scale > 1.0 else "samples"
                log.update({"original_sample": original_log[original_sample_key]})

            out[1]["c_crossattn"] = adjust_crossattn
            inputs = [out, idx, x_T]
            sample_log, idx = super().log_images(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                                 unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
            log.update(sample_log)
            return log, idx
        else:
            return super().log_images(batch=batch, N=N, return_inputs=return_inputs,
                                      unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)