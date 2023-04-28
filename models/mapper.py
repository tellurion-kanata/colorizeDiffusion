import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from utils import instantiate_from_config
from ldm.modules.encoders.modules import OpenCLIP, OpenCLIPEncoder
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import gap, CondTransformer


def exists(v):
    return v is not None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def calculate_scale(v: torch.Tensor, t: torch.Tensor, logit_scale_exp: torch.Tensor):
    """
        Calculate the projection of v along the direction of t
        params:
            v: visual features predicted by clip image encoder, shape: (b, n, c)
            t: text feature predicted by clip text encoder, shape: (b, c)
    """
    image_features = v / v.norm(dim=2, keepdim=True)
    text_features = t / t.norm(dim=1, keepdim=True)

    proj = image_features @ text_features.t() * logit_scale_exp
    t_square = (text_features ** 2).sum(dim=1).unsqueeze(0)
    return proj / t_square


class ReferenceWrapper(nn.Module):
    def __init__(self,
                 clip_config,
                 pool_config=None,
                 drop_rate=0.5
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
        if self.training:
            z = torch.bernoulli((1 - self.drop_rate) * torch.ones(z.shape[0], device=z.device)[:, None, None]) * z
        return  {"c_crossattn": [z.detach()]}


class ConditionWrapper(nn.Module):
    def __init__(self,
                 clip_config: dict,
                 n_emb=512,
                 emb_dims=1024,
                 pool_config=None,
                 use_codebook=False,
                 drop_rate=0.,
                 use_adm=False,
                 ):
        super().__init__()
        self.encoder = OpenCLIPEncoder(**clip_config)
        self.latent_pooling = instantiate_from_config(pool_config) if exists(pool_config) else None
        self.priorbook = nn.Parameter(torch.randn([n_emb, emb_dims])) if use_codebook else None

        self.use_adm = (clip_config.get(type, "pooled") == "pooled" and use_adm)
        self.n_emb = n_emb
        self.drop_rate = drop_rate
        self.sample = False

    def get_input(self, batch, device):
        x, r = batch["sketch"], batch["reference"]
        x, r = map(lambda t: t.to(memory_format=torch.contiguous_format).to(device), (x, r))
        self.sample = batch.get("sample", False)
        return {"sketch": x, "reference": r}

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
            z = self.latent_pooling(z, self.sample)

        # drop reference conditions according to the drop_rate
        if self.training and self.drop_rate:
            z = torch.bernoulli((1 - self.drop_rate) * torch.ones(z.shape[0], device=z.device)[:, None, None]) * z

        c_dict = {"c_concat": [s]}
        # using cross-attention path for visual tokens
        if not self.use_adm:
            if exists(self.priorbook):
                priorbook = torch.ones([z.shape[0]], device=z.device)[:, None, None] * self.priorbook
                z = torch.cat([z, priorbook], dim=1)
            c_dict.update({"c_crossattn": [z]})
        else:
            c_dict.update({"c_adm": [gap(z, keepdim=False)]})
        return c_dict

    def get_unconditional_conditioning(self, c, label="reference"):
        assert label in ["sketch", "reference"]
        if label == "reference":
            crossattn = c["c_crossattn"][0]
            uc = {"c_concat": c["c_concat"], "c_crossattn": [torch.zeros_like(crossattn, device=crossattn.device)]}
        else:
            concat = c["c_concat"][0]
            uc = {"c_concat": [torch.zeros_like(concat, device=concat.device)], "c_crossattn": c["c_crossnattn"]}
        return uc


class PromptMapper(pl.LightningModule):
    def __init__(self, diffusion_config, clip_config, mapper_config, lossconfig, offset=1, type="pooled"):
        super().__init__()
        assert type in ["pooled", "tokens"]
        self.type = type
        self.offset = offset
        self.build_diffusion(diffusion_config)

        self.clip = OpenCLIP(**clip_config)
        self.mapper = PromptMapper(**mapper_config)
        self.loss = instantiate_from_config(lossconfig)

    def build_diffusion(self, config):
        self.diffusion = LatentDiffusion(**config).eval()
        self.diffusion.train = disabled_train
        del self.diffusion.cond_stage_model

        for param in self.diffusion.parameters():
            param.requires_grad = False

    def get_input(self, batch, return_first_stage_outputs=False,
                  return_original_cond=False, bs=None, return_x=False):
        if bs:
            for k in batch:
                batch[k] = batch[k][:bs]

        x = batch["color"]
        ref = batch["reference"]
        ske = batch["sketch"]
        text = batch["text"]
        idx = batch["index"]

        z = self.diffusion.get_first_stage_encoding(self.diffusion.encode_first_stage(x)).detach()
        c_crossattn = self.clip.encode_image(ref)
        text_features = self.clip.encode_text(text)

        out = [z, ske, c_crossattn, text_features]
        if return_first_stage_outputs:
            xrec = self.diffusion.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append({"sketch": ske, "reference": ref})
        return out, idx

    def forward(self, image_features, text_features, scale):
        predict_image_features = self.mapper(image_features, text_features, scale)
        return predict_image_features

    def get_scale(self, v, t):
        """
            Shift the visual features forward to get a set of incorrect image features.
            When adopting tokens as reference visual features, the scale would be (b, n, 1
        """
        shifted_v = torch.roll(v, self.offset)
        correct_scale = calculate_scale(v, t, self.clip.logit_scale_exp).mean(dim=1, keepdims=True)
        shifted_scale = calculate_scale(shifted_v, t, self.clip.logit_scale_exp)
        dscale = correct_scale - shifted_scale
        return shifted_v, correct_scale, dscale

    def training_step(self, batch, batch_idx):
        out, idx = self.get_input(batch)
        x, c_concat, image_features, text_features = out

        shifted_features, scale, dscale = self.get_scale(image_features, text_features)
        fake_features = self(shifted_features, text_features, dscale)
        loss, loss_dict = self.loss(x, c_concat, fake_features, image_features, shifted_features,
                                    text_features, scale, self.clip.logit_scale_exp, self.diffusion)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        opt = optim.AdamW(self.mapper.parameters(), lr=self.lr)
        return opt

    def log_images(self, batch, scale=None, **kwargs):
        out, idx = self.get_input(batch, return_first_stage_outputs=True, return_original_cond=True)
        z, c_concat, image_features, text_features, x, xrec, xc = out

        if exists(scale):
            cscale = calculate_scale(image_features, text_features, self.clip.logit_scale_exp)
            dscale = scale - cscale
        else:
            # sampling during training and validation
            image_features, _, dscale = self.get_scale(image_features, text_features)
        c_crossattn = self(image_features, text_features, dscale)

        c = {"c_concat": c_concat, "c_crossattn": c_crossattn}
        inputs = [[z, c, x, xrec, xc], idx]
        log, idx = self.diffusion.log_images(batch=None, inputs=inputs, **kwargs)
        return log, idx