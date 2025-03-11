import math
import open_clip

from tqdm import tqdm
from einops import rearrange
from refnet.util import exists, append_dims
from refnet.sampling import tps_warp
from refnet.ldm.openaimodel import Timestep, zero_module

import timm
import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


versions = {
    "ViT-bigG-14": "laion2b_s39b_b160k",
    "ViT-H-14": "laion2b_s32b_b79k",        # resblocks layers: 32
    "ViT-L-14": "laion2b_s32b_b82k",
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384": None,       # arch name [DFN-ViT-H]
}
cache_dir = "./pretrained_models"



class OpenCLIP(nn.Module):
    def __init__(self, vision_config=None, text_config=None, **kwargs):
        super().__init__()
        if exists(vision_config):
            vision_config.update(kwargs)
        else:
            vision_config = kwargs

        if exists(text_config):
            text_config.update(kwargs)
        else:
            text_config = kwargs

        self.visual = FrozenOpenCLIPImageEmbedder(**vision_config)
        self.transformer = FrozenOpenCLIPEmbedder(**text_config)

    def preprocess(self, x):
        return self.visual.preprocess(x)

    @property
    def scale_factor(self):
        return self.visual.scale_factor

    def update_scale_factor(self, scale_factor):
        self.visual.update_scale_factor(scale_factor)

    def encode(self, *args, **kwargs):
        return self.visual.encode(*args, **kwargs)

    @torch.no_grad()
    def encode_text(self, text, normalize=True):
        return self.transformer(text, normalize)

    def calculate_scale(self, v: torch.Tensor, t: torch.Tensor):
        """
            Calculate the projection of v along the direction of t
            params:
                v: visual tokens from clip image encoder, shape: (b, n, c)
                t: text features from clip text encoder (argmax -1), shape: (b, 1, c)
        """
        return v @ t.mT



class FrozenOpenCLIPImageEmbedder(nn.Module):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """
    def __init__(
        self,
        scale_factor = 1.,
        arch = "ViT-H-14",
        device = "cuda",
        freeze = True,
        antialias = True,
        repeat_to_max_len = False,
        num_image_crops = 0,
        layer_idx = 0,
        normalize = True,
        *args,
        **kwargs
    ):
        super().__init__()

        if arch == "DFN-ViT-H":
            arch = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
            self.image_size = 378
        else:
            self.image_size = 224

        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=versions[arch],
            cache_dir=cache_dir
        )
        del model.transformer
        self.model = model.visual
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.layer_idx = layer_idx
        self.normalize = normalize
        self.device = device
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1),
                             persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1),
                             persistent=False)

        self.scale_factor = scale_factor
        self.adjust_positional_embedding(scale_factor)
        tqdm.write("Initialize CLIP image encoder.")


    def update_scale_factor(self, scale_factor):
        tqdm.write("Update the length of positional embedding")
        self.scale_factor = scale_factor
        self.adjust_positional_embedding(scale_factor)

    def preprocess(self, x,):
        # normalize to [0,1]
        ns = int(self.image_size * self.scale_factor)
        x = F.interpolate(x, (ns, ns), mode="bicubic", align_corners=True, antialias=self.antialias)
        x = (x + 1.0) / 2.0

        # renormalize according to clip
        x = (x - self.mean) / self.std
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def interpolate_positional_embedding(self, x: torch.Tensor, scale_factor, mode="bicubic"):
        n, c = x.shape
        h = w = int(math.sqrt(n))
        x = x.unsqueeze(0).permute(0, 2, 1).view(1, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        x = x.squeeze(0).view(c, int(n*scale_factor*scale_factor)).permute(1, 0)
        return x

    def adjust_positional_embedding(self, scale_factor):
        if scale_factor > 1:
            positional_embedding = torch.Tensor(self.model.positional_embedding[1:])
            positional_embedding = self.interpolate_positional_embedding(positional_embedding, scale_factor)
            class_positional_embedding = self.model.positional_embedding[0].unsqueeze(0)
            positional_embedding = torch.cat([class_positional_embedding, positional_embedding], dim=0)
        else:
            positional_embedding = torch.Tensor(self.model.positional_embedding)
        positional_embedding = positional_embedding.to(self.model.conv1.weight.device)
        self.positional_embedding = positional_embedding

    def visual_tokenizer(self, x):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                  device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype).to(x.device)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.patch_dropout(x)
        x = self.model.ln_pre(x)
        return x

    def post_process(self, x, output):
        pooled, tokens = self.model._global_pool(x)
        if output == "full":
            outputs = x
        elif output == "local":
            outputs = tokens
        else:
            outputs = pooled.unsqueeze(1)

        if self.normalize:
            outputs = self.model.ln_post(outputs)
            outputs = outputs @ self.model.proj
        return outputs

    def forward(self, x, output="cls"):
        x = self.visual_tokenizer(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.post_process(x, output)
        return x


    def transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    @torch.no_grad()
    def encode(self, img, output_type="cls", preprocess=True, warp_p=0.):
        img = self.preprocess(img) if preprocess else img

        if warp_p > 0.:
            rand = append_dims(torch.rand(img.shape[0], device=img.device, dtype=img.dtype), img.ndim)
            img = torch.where(torch.Tensor(rand > warp_p), img, tps_warp(img))
        return self(img, output_type)



class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(
            self,
            arch="ViT-bigG-14",
            device="cuda",
            max_length=77,
            freeze=True,
            clip_skip=1,
            *args,
            **kwargs
    ):
        super().__init__()
        if arch == "DFN-ViT-H":
            arch = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"

        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device = torch.device('cpu'),
            pretrained = versions[arch],
            cache_dir = cache_dir
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.clip_skip = clip_skip

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, normalize=True):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device), normalize)
        return z

    def encode_with_transformer(self, text, projection=True, normalize=False):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        if projection:
            x = x @ self.model.text_projection

        if normalize:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            x = x / x.norm(dim=1, keepdim=True)
            x = x.unsqueeze(1)

        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.clip_skip:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    @torch.no_grad()
    def encode(self, text):
        return self(text)


class ScalarEmbedder(nn.Module):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.timestep = Timestep(embed_dim)
        self.embed_layer = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            zero_module(nn.Linear(out_dim, out_features=out_dim))
        )

    def forward(self, x, dtype=torch.float32):
        emb = self.timestep(x)
        emb = rearrange(emb, "b d -> b 1 d")
        emb = self.embed_layer(emb.to(dtype))
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.timestep = Timestep(embed_dim)

    def forward(self, x):
        x = self.timestep(x)
        return x


if __name__ == '__main__':
    import PIL.Image as Image

    encoder = FrozenOpenCLIPImageEmbedder(arch="DFN-ViT-H")
    image = Image.open("../../miniset/origin/70717450.jpg").convert("RGB")
    image = (torchvision.transforms.ToTensor()(image) - 0.5) * 2
    image = image.unsqueeze(0)
    print(image.shape)
    feat = encoder.encode(image, "local")
    print(feat.shape)