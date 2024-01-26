from sgm.models.diffusion import DiffusionEngine
from reflora.modules.lora import RefLoraModules



class RefNet(RefLoraModules, DiffusionEngine):
    def __init__(self, r, ckpt_path, init_from_sd=True, **kwargs):
        pretrained_sd_ckpt = ckpt_path if init_from_sd else None
        super().__init__(ckpt_path=pretrained_sd_ckpt, **kwargs)

        self.r = r
        self.is_xl = True
        self.inject_lora_weights()
        if not init_from_sd:
            self.init_from_ckpt(ckpt_path)