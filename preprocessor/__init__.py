import torch
import torch.nn as nn
import os.path as osp
import functools

from basicsr.utils.download_util import load_file_from_url
from .anime2sketch import UnetGenerator
from .manga_line_extractor import res_skip
from .sketchKeras import SketchKeras


remote_model_dict = {
    "lineart": "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth",
    "lineart_denoise": "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth",
    "lineart_keras": "https://huggingface.co/tellurion/line_extractor/resolve/main/model.pth",
}
model_path = "preprocessor/weights"


def create_model(model="lineart"):
    """Create a model for anime2sketch
    hardcoding the options for simplicity
    """
    assert model in remote_model_dict.keys()
    remote_path = remote_model_dict[model]
    basename = osp.basename(remote_path)
    ckpt_path = osp.join(model_path, basename)

    if not osp.exists(ckpt_path):
        load_file_from_url(remote_path, model_dir=model_path)

    if model == "lineart":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    elif model == "lineart_denoise":
        net = res_skip()
    elif model == "lineart_keras":
        net = SketchKeras()
    else:
        return None

    ckpt = torch.load(ckpt_path)
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    net.load_state_dict(ckpt)
    return net.eval()