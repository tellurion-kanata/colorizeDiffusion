import torch
import torch.optim as optim
import pytorch_lightning as pl

from ldm.models.diffusion.ddpm import LatentDiffusion, exists
from ldm.modules.encoders.modules import OpenCLIP, disabled_train
from loss import MappingLoss




class PromptMapper(pl.LightningModule):
    def __init__(self, diffusion_config, mapper_config=None, clip_config=None, lossconfig=None, offset=1, type="pooled"):
        super().__init__()
        assert type in ["pooled", "tokens"]
        self.type = type
        self.offset = offset
        self.build_diffusion(diffusion_config)

        self.clip = OpenCLIP(**clip_config if exists(clip_config) else None)
        self.mapper = PromptMapper(**mapper_config if exists(mapper_config) else None)
        self.loss = MappingLoss(**lossconfig if exists(lossconfig) else None)

    def build_diffusion(self, config):
        self.diffusion = LatentDiffusion(**config).eval()
        self.diffusion.train = disabled_train

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
            When adopting tokens as reference visual features, the scale would be (b, n, 1)
        """
        shifted_v = torch.roll(v, self.offset)
        correct_scale = self.clip.calculate_scale(v, t) * v / v.mean(dim=1, keepdims=True)
        shifted_scale = self.clip.calculate_scale(shifted_v, t) * shifted_v / shifted_v.mean(dim=1, keepdims=True)
        dscale = correct_scale - shifted_scale
        return shifted_v, correct_scale, dscale

    def training_step(self, batch, batch_idx):
        out, idx = self.get_input(batch)
        x, c_concat, image_features, text_features = out

        shifted_features, scale, dscale = self.get_scale(image_features, text_features)
        fake_features = self(shifted_features, text_features, dscale)
        loss, loss_dict = self.loss(x, c_concat, fake_features, image_features, shifted_features,
                                    text_features, scale, self.clip.calculate_scale, self.diffusion)

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
            cscale = self.clip.calculate_scale(image_features, text_features)
            dscale = scale - cscale
        else:
            # sampling during training and validation
            image_features, _, dscale = self.get_scale(image_features, text_features)
        c_crossattn = self(image_features, text_features, dscale)

        c = {"c_concat": c_concat, "c_crossattn": c_crossattn}
        inputs = [[z, c, x, xrec, xc], idx]
        log, idx = self.diffusion.log_images(batch=None, inputs=inputs, **kwargs)
        return log, idx