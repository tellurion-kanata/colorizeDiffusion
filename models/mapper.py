import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import xformers

from ldm.models.diffusion.ddpm import exists
from ldm.modules.attention import gap, MemoryEfficientCrossAttention
from ldm.modules.encoders.modules import OpenCLIP, disabled_train
from utils import instantiate_from_config
from models.loss import MappingLoss



class PromptMapper(pl.LightningModule):
    def __init__(self, diffusion_config, context_dim, clip_config={}, offset=1, type="tokens"):
        super().__init__()
        assert type in ["pooled", "tokens"]
        self.type = type
        self.offset = offset
        self.build_diffusion(diffusion_config)

        self.clip = OpenCLIP(**clip_config)
        self.M = nn.Linear(context_dim, context_dim)
        # self.mapper = PromptTransformer(**mapper_config)
        self.loss = MappingLoss()

    def build_diffusion(self, config):
        self.diffusion = instantiate_from_config(config).eval()
        self.diffusion.train = disabled_train

    def init_from_ckpt(self, path, ignore_keys={}):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def get_input(self, batch, return_first_stage_outputs=False, text=None,
                  return_original_cond=False, bs=None, return_x=False):
        if bs:
            for k in batch:
                if k == "sample":
                    continue
                batch[k] = batch[k][:bs]

        x = batch["color"]
        ref = batch["reference"]
        ske = batch["sketch"]
        idx = batch["index"]
        text = batch["text"] if not exists(text) else [text] * x.shape[0]

        z = self.diffusion.get_first_stage_encoding(self.diffusion.encode_first_stage(x)).detach()
        c_crossattn = self.clip.encode_image(ref)
        # text_features, arg_text_features = self.clip.encode_text(text)
        arg_text_features = self.clip.encode_text(text)

        # out = [z, ske, c_crossattn, text_features, arg_text_features]
        out = [z, ske, c_crossattn, arg_text_features]
        if return_first_stage_outputs:
            xrec = self.diffusion.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append({"sketch": ske, "reference": ref})
        return out, idx

    def forward(self, v: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        t = t.repeat(1, v.shape[1], 1)
        gap_v = gap(v)
        gap_v = torch.where(gap_v != 0, gap_v, torch.full_like(gap_v, 1e-6))
        fake_v = v + t * s * self.mapper(v/gap_v)
        # fake_v = self.mapper(v, t, s)
        return fake_v

    def get_scale(self, v, t):
        """
            Shift the visual features forward to get a set of incorrect image features.
            When adopting tokens as reference visual features, the scale would be (b, n, 1)
        """
        shifted_v = torch.roll(v, self.offset, dims=0)
        correct_scale = self.clip.calculate_scale(gap(v), t)
        shifted_scale = self.clip.calculate_scale(gap(shifted_v), t)
        return shifted_v, correct_scale - shifted_scale

    def training_step(self, batch, batch_idx):
        out, idx = self.get_input(batch)
        # x, c_concat, image_features, text_features, arg_text_features = out
        x, c_concat, image_features, arg_text_features = out

        shifted_features, dscale = self.get_scale(image_features, arg_text_features)
        fake_features = self(shifted_features, arg_text_features, dscale)
        loss, loss_dict = self.loss(x, c_concat, fake_features, image_features, shifted_features,
                                    arg_text_features, self.clip.calculate_scale, self.diffusion)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        opt = optim.AdamW(self.mapper.parameters(), lr=self.lr)
        return opt

    def log_images(self, batch, target_scale=None, N=8, control=None, target=None, return_inputs=True,
                   sample_original_cond=True, unconditional_guidance_scale=1.0, **kwargs):
        out, idx = self.get_input(batch, bs=N, return_first_stage_outputs=True, return_original_cond=True,
                                  text=target)
        log = {}
        # z, c_concat, image_features, text_features, arg_text_features, x, xrec, xc = out
        z, c_concat, image_features, arg_text_features, x, xrec, xc = out
        if exists(target_scale):
            scale = self.clip.calculate_scale(image_features, arg_text_features)
            global_scale = gap(scale)
            print(f"current global scale: {global_scale}")
            dscale = target_scale - global_scale
        else:
            # sampling during training
            image_features, dscale = self.get_scale(image_features, arg_text_features)

        x_T = torch.randn_like(z, device=z.device)
        if sample_original_cond:
            c = {"c_concat": [c_concat], "c_crossattn": [image_features]}
            inputs = [[z, c], idx, x_T]
            original_log, _ = self.diffusion.log_images(batch=None, inputs=inputs, N=N, return_inputs=False,
                                                        unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
            original_sample_key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}" if unconditional_guidance_scale > 1.0 else "samples"
            log.update({"original_sample": original_log[original_sample_key]})

        # c_crossattn = self(image_features, text_features, dscale)
        c_crossattn = self(image_features, arg_text_features, dscale)

        c = {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}
        inputs = [[z, c, x, xrec, xc], idx, x_T] if return_inputs else [[z, c], idx, x_T]
        sample_log, idx = self.diffusion.log_images(batch=None, inputs=inputs, N=N, return_inputs=return_inputs,
                                                    unconditional_guidance_scale=unconditional_guidance_scale, **kwargs)
        log.update(sample_log)
        return log, idx