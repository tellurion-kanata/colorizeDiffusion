from ldm.models.diffusion.ddpm import LatentDiffusion
from utils import instantiate_from_config

class CondDiffGAN(LatentDiffusion):
    def __init__(self, lossconfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = instantiate_from_config(lossconfig)

    def training_step(self, batch, batch_idx, optimizer_idx):

    def configure_optimizers(self):
