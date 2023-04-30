import torch
import torch.nn as nn


from utils import instantiate_from_config
from ldm.modules.encoders.modules import OpenCLIPEncoder
from ldm.modules.attention import gap


def exists(v):
    return v is not None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


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