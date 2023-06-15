import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as transforms

from utils import instantiate_from_config
from ldm.modules.encoders.modules import OpenCLIPEncoder, OpenCLIP
from ldm.models.diffusion.ddpm import LatentDiffusion

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

def normalize(img, image_size=512):
    img = transforms.Resize((image_size, image_size))(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5), (0.5))(img)
    return img.unsqueeze(0)

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
                 pool_config=None,
                 init_dr=0.5,
                 finl_dr=0.5,
                 decl_dr=0.,
                 use_adm=False,
                 encoder_type="image",
                 ):
        super().__init__()
        self.encoder = self.OpenCLIPEncoders[encoder_type](**clip_config)
        self.latent_pooling = instantiate_from_config(pool_config) if exists(pool_config) else None

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

    def encode_img(self, img, resolution):
        return self.encoder.encode(self.encoder.preprocess(img, resolution=resolution))

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

        c_dict = {"c_concat": [s], "c_crossattn": [z]}
        return c_dict


import numpy as np
class AdjustLatentDiffusion(LatentDiffusion):
    def __init__(self, type="tokens", *args, **kwargs):
        assert type in ["tokens", "global", "full"]
        super().__init__(*args, **kwargs)
        self.type = type

    def compute_pwm(self, s: torch.Tensor, dscale: torch.Tensor, ratio=2, thresholds=[0.5, 0.55, 0.65, 0.95]):
        """
            The shape of input scales tensor should be (b, n, 1)
        """
        assert len(s.shape) == 3, len(thresholds) == 4
        maxm = s.max(dim=1, keepdim=True).values
        minm = s.min(dim=1, keepdim=True).values
        d = maxm - minm

        maxmin = (s - minm) / d

        adjust_scale = torch.where(maxmin <= thresholds[0],
                                   -dscale * ratio,
                                   -dscale + dscale * (maxmin - thresholds[0]) / (thresholds[1]-thresholds[0]))
        adjust_scale = torch.where(maxmin > thresholds[1],
                                   0.5 * dscale * (maxmin-thresholds[1]) / (thresholds[2] - thresholds[1]),
                                   adjust_scale)
        adjust_scale = torch.where(maxmin > thresholds[2],
                                   0.5 * dscale + 0.5 * dscale * (maxmin - thresholds[2]) / (thresholds[3] - thresholds[2]),
                                   adjust_scale)
        adjust_scale = torch.where(maxmin > thresholds[3], dscale, adjust_scale)
        return adjust_scale

    def manipulate_step(self, v, t, target_scale, c=None, enhance=False, thresholds=[]):
        if self.type == "global":
            t = [t] * v.shape[0]
            t = self.cond_stage_model.encode_text(t)

            if c != "None":
                c = [c] * v.shape[0]
                c = self.cond_stage_model.encode_text(c)
                if enhance:
                    s_c = self.cond_stage_model.calculate_scale(v, c)
                    v = v - s_c * c
                else:
                    v = v + target_scale * (t - c)
                    return v
            cur_target_scale = self.cond_stage_model.calculate_scale(v, t)
            print(f"current target scale: {cur_target_scale}")
            v = v + (target_scale - cur_target_scale) * t
        else:
            cls_token = v[:, 0].unsqueeze(1)
            v = v[:, 1:]
            c = [c] * v.shape[0]
            t = [t] * v.shape[0]
            c = self.cond_stage_model.encode_text(c)
            t = self.cond_stage_model.encode_text(t)

            control_scale = self.cond_stage_model.calculate_scale(cls_token, c)
            cur_target_scale = self.cond_stage_model.calculate_scale(cls_token, t)
            dscale = target_scale - cur_target_scale
            print(f"current global target scale: {cur_target_scale}, global control scale: {control_scale}")

            c_map = self.cond_stage_model.calculate_scale(v, c)
            pwm = self.compute_pwm(c_map, dscale, thresholds=thresholds)
            base = 1 if enhance else 0
            v = v + (pwm + base * c_map) * (t - c)

            v = torch.cat([cls_token, v], dim=1)
        return v

    def manipulate(self, v, target, target_scales, control=None, enhance_list=[], thresholds_list=[]):
        """
            v: visual tokens in shape (b, n, c)
            target: target text embeddings in shape (b, 1 ,c)
            control: control text embeddings in shape (b, 1, c)
        """
        if self.type == "global":
            for t, c, s_t, enhance in zip(target, control, target_scales, enhance_list):
                v = self.manipulate_step(v, t, s_t, c, enhance)
        else:
            for t, c, s_t, enhance, thresholds in zip(target, control, target_scales, enhance_list, thresholds_list):
                v = self.manipulate_step(v, t, s_t, c, enhance, thresholds)
        return [v]


    def log_images(self, batch, N=8, control=[], target=[], target_scale=[], thresholds=[[0.5, 0.55, 0.65, 0.95]], is_train=False,
                   return_inputs=True, sample_original_cond=True, unconditional_guidance_scale=1.0, enhance=[], **kwargs):
        def sample(inputs, sample_function):
            original_log, _ = sample_function(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                              unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
            original_sample_key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}" \
                if unconditional_guidance_scale > 1.0 else "samples"
            return original_log[original_sample_key]

        if len(target) > 0:
            assert len(target) == len(target_scale), "Each prompt should have a target scale"
            out, idx = super().get_input(batch, self.first_stage_key,
                                         return_first_stage_outputs=return_inputs,
                                         cond_key=self.cond_stage_key,
                                         force_c_encode=True,
                                         return_original_cond=return_inputs,
                                         bs=N)
            z, c = out[:2]
            v = c["c_crossattn"][0]
            adjust_v = self.manipulate(v, target, target_scale, control, enhance_list=enhance, thresholds_list=thresholds)
            if self.type == "tokens":
                out[1]["c_crossattn"] = [v[:, 1:]]
                adjust_v = [adjust_v[0][:, 1:]]

            log = {}
            x_T = torch.randn_like(z, device=z.device)
            if sample_original_cond:
                                # if self.type == "tokens":
                #     out[1]["c_crossattn"] = [v]
                log.update({"original_sample_v": sample([out, idx, x_T], super().log_images)}, )

            out[1]["c_crossattn"] = adjust_v
            inputs = [out, idx, x_T]
            sample_log, idx = super().log_images(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                                 unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
            log.update(sample_log)
            return log, idx
        else:
            return super().log_images(batch=batch, N=N, return_inputs=return_inputs,
                                      unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)

    """
        The following part is for User interface. 
    """
    def get_heatmap(self, s: torch.Tensor, threshold: float):
        """
            The shape of input scales tensor should be (b, n, 1)
        """
        maxm = s.max(dim=1, keepdim=True).values
        minm = s.min(dim=1, keepdim=True).values
        d = maxm - minm
        return torch.where((s - minm) / d < threshold, torch.zeros_like(s), torch.ones_like(s) * 0.25)

    def get_tokens(self, image, resolution=224):
        return self.cond_stage_model.encode_img(image, resolution)

    def get_projections(self, v, t):
        return self.cond_stage_model.calculate_scale(v, t)

    """
        User interactive function.
    """
    @torch.no_grad()
    def generate_image(self, sketch, reference, unconditional_guidance_scale, resolution,
                       ddim_steps, ddim_eta, use_ema_scope, enhance=[], control=[], target=[],
                       thresholds_list=[], target_scales=[], seed=None):
        # set global seed for generationz
        pl.seed_everything(seed)

        # modify positional embedding according to image resolution
        self.cond_stage_model.encoder.adjust_scale_factor(int(resolution))
        v = self.cond_stage_model.encode_img(reference, int(resolution/512*224))

        # manipulate reference image embeddings
        if target_scales[0] > 0.:
            assert len(target) == len(target_scales)
            v = self.manipulate(v, target, target_scales, control, enhance, thresholds_list)[0]
        if self.type == "tokens":
            v = v[:, 1:]

        # generate images
        sketch = normalize(sketch, image_size = resolution).to(self.device)
        c = {"c_concat": [sketch], "c_crossattn": [v]}
        inputs = [[None, c], None, None]
        logs, _ = super().log_images(batch=None, inputs=inputs, return_inputs=False, ddim_steps=ddim_steps, ddim_eta=ddim_eta, 
        					sample=unconditional_guidance_scale==1., use_ema_scope=use_ema_scope, 
        					unconditional_guidance_scale=unconditional_guidance_scale, unconditional_guidance_label="reference")
        sample_key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}" if unconditional_guidance_scale > 1.0 else "samples"

        # convert to RGB format
        img = torch.clamp(logs[sample_key][0], -1., 1.)
        img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        img = img.permute(1, 2, 0).squeeze(-1)
        img = img.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img
