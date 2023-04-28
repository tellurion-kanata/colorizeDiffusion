import torch
import torch.optim as optim
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils import instantiate_from_config


class Autoencoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 image_key='color',
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 scheduler_config=None,
                 ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.image_key = image_key

        self.loss = instantiate_from_config(lossconfig)
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config

    def init_from_ckpt(self, path, ignore_keys=dict()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, bs=None):
        x = batch[self.image_key].to(memory_format=torch.contiguous_format).float()
        idx = batch['index']

        if bs:
            return x[:bs], idx[:bs]
        return x, idx

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        fake = self.decoder(z)
        return fake

    def forward(self, x, sample_posterior=True):
        z = self.encode(x)
        fake = self.decode(z)
        return fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = self.get_input(batch)
        fake = self(x)

        if optimizer_idx == 0:
            g_loss, g_loss_dict = self.loss(x, fake, optimizer_idx, self.global_step)
            self.log("rec_loss", g_loss_dict['train/rec_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(g_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return g_loss

        if optimizer_idx == 1:
            d_loss, d_loss_dict = self.loss(x, fake, optimizer_idx, self.global_step)
            self.log("d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(d_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return d_loss


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        lr = self.lr
        optim_G = optim.Adam(list(self.encoder.parameters()) +
                             list(self.decoder.parameters()), lr=lr, betas=(0.5, 0.99))
        optim_D = optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.99))
        return [optim_G, optim_D], []

    @torch.no_grad()
    def log_images(self, batch, bs=None, **kwargs):
        log = dict()
        x, idx = self.get_input(batch, bs)
        log['color'] = x
        log['reconstructions'] = self(x)
        return log, idx


class VAE(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 image_key='color',
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 scheduler_config=None,
                 ):
        super(VAE, self).__init__()
        embed_dim = ddconfig['z_channels']
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, 2 * embed_dim, kernel_size=1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.image_key = image_key
        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config

    def init_from_ckpt(self, path, ignore_keys=dict()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, bs=None):
        x = batch[self.image_key].to(memory_format=torch.contiguous_format).float()
        idx = batch['index']

        if bs:
            return x[:bs], idx[:bs]
        return x, idx

    def encode(self, x):
        z = self.encoder(x)
        moments = self.quant_conv(z)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        fake = self.decoder(z)
        return fake

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        fake = self.decode(z)
        return fake, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = self.get_input(batch)
        fake, posterior = self(x)

        if optimizer_idx == 0:
            g_loss, g_loss_dict = self.loss(x, fake, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer())
            self.log("rec_loss", g_loss_dict['train/rec_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(g_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return g_loss

        if optimizer_idx == 1:
            d_loss, d_loss_dict = self.loss(x, fake, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer())
            self.log("d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(d_loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return d_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def configure_optimizers(self):
        lr = self.lr
        optim_G = optim.Adam(list(self.encoder.parameters()) +
                             list(self.decoder.parameters()) +
                             list(self.quant_conv.parameters()) +
                             list(self.post_quant_conv.parameters()), lr=lr, betas=(0.5, 0.9))
        optim_D = optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [optim_G, optim_D], []

    @torch.no_grad()
    def log_images(self, batch, bs=None, **kwargs):
        log = dict()
        x, idx = self.get_input(batch, bs)

        log['color'] = x
        log['reconstructions'], posterior = self(x)
        log['samples'] = self.decoder(torch.randn_like(posterior.mode()))
        return log, idx
