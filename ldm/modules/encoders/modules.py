import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import open_clip
import os

versions = {
    "ViT-bigG-14": "laion2b_s39b_b160k",
    "ViT-H-14": "laion2b_s32b_b79k"
}

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class OpenCLIP(nn.Module):
    def __init__(self,
                 arch="ViT-H-14",
                 device="cuda",
                 type='pooled',
                 cache_dir="./pretrained_models",
                 freeze=True,
                 layer="last",
                 use_positional_embedding=False,
                 ):
        super().__init__()
        pretrained_version = versions[arch]
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained_version,
                                                            cache_dir=cache_dir)

        self.visual = OpenCLIPEncoder(model=model, freeze=freeze, device=device, type=type,
                                      use_positional_embedding=use_positional_embedding)
        self.transformer = FrozenOpenCLIPEmbedder(model=model, freeze=freeze, layer=layer)
        self.logit_scale_exp = model.logit_scale.exp()

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.transformer(text)

    def calculate_scale(self, v: torch.Tensor, t: torch.Tensor):
        """
            Calculate the projection of v along the direction of t
            params:
                v: visual tokens predicted by clip image encoder, shape: (b, n, c)
                t: text feature predicted by clip text encoder (argmax -1), shape: (b, c)
        """
        image_features = v / v.norm(dim=2, keepdim=True)
        text_features = t / t.norm(dim=2, keepdim=True)

        text_features = text_features.permute(0, 2, 1)
        proj = torch.bmm(image_features, text_features)
        t_square = (text_features ** 2).sum(dim=1, keepdim=True)
        return proj / t_square


class OpenCLIPEncoder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for image
     options:
        arch: 'ViT-H-14', version: 'laion2b_s32b_b79k', dims: 1024
        arch: 'ViT-bigG-14', version: 'laion2b_s39b_b160k', dims: 1280
    """

    def __init__(self,
                 scale_factor=1.,
                 model=None,
                 arch="ViT-H-14",
                 device="cuda",
                 type="pooled",
                 cache_dir="./pretrained_models",
                 freeze=True,
                 use_positional_embedding=False,
                 ):
        super().__init__()
        assert type in ["pooled", "tokens"]
        pretrained_version = versions[arch]
        if model is None:
            model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained_version, cache_dir=os.path.abspath(cache_dir))
            del model.transformer
        self.model = model.visual

        if type == "pooled":
            scale_factor = 1.
            self.output_tokens = False
        elif type == "tokens":
            self.output_tokens = True
        else:
            raise NotImplementedError()

        if use_positional_embedding:
            if scale_factor > 1.:
                positional_embedding = self.model.positional_embedding[:-1]
                positional_embedding = self.interpolate_positional_embedding(positional_embedding, scale_factor, "bicubic")
                class_positional_embedding = self.model.positional_embedding[-1].unsqueeze(0)
                positional_embedding = torch.cat([positional_embedding, class_positional_embedding], dim=0)
            else:
                positional_embedding = self.model.positional_embedding
            self.positional_embedding = positional_embedding
            print("Adopting positional embedding in Vision Transformer.")
        else:
            self.positional_embedding = None

        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

    def interpolate_positional_embedding(self, x: torch.Tensor, scale_factor, mode):
        n, c = x.shape
        h = w = int(math.sqrt(n))
        x = x.unsqueeze(0).permute(0, 2, 1).view(1, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        x = x.squeeze(0).view(c, int(n*scale_factor*scale_factor)).permute(1, 0)
        return x

    def transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def forward(self, x: torch.Tensor):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.to(x.dtype).to(x.device)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.patch_dropout(x)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled, tokens = self.model._global_pool(x)
        output = tokens if self.output_tokens else pooled.unsqueeze(1)
        output = self.model.ln_post(output)
        output = output @ self.model.proj
        return output

    def encode(self, image):
        return self(image.to(self.device))


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-bigG-14", version="laion2b_s39b_b160k", device="cuda", max_length=77,
                 freeze=True, layer="last", model=None):
        super().__init__()
        assert layer in self.LAYERS
        if model is None:
            model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
            del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x.unsqueeze(1)

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)