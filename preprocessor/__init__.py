import os

import torch.hub
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import functools

model_path = "preprocessor/weights"
os.environ["HF_HOME"] = model_path
torch.hub.set_dir(model_path)

from torch.hub import download_url_to_file
from transformers import AutoModelForImageSegmentation
from .anime2sketch import UnetGenerator
from .manga_line_extractor import res_skip
from .sketchKeras import SketchKeras
from .sk_model import LineartDetector
from .anime_segment import ISNetDIS
from util import load_weights
# from .lang_sam import LangSAM


remote_model_dict = {
    "lineart": "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth",
    "lineart_denoise": "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth",
    "lineart_keras": "https://huggingface.co/tellurion/line_extractor/resolve/main/model.pth",
    "lineart_sk": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
    "ISNet": "https://huggingface.co/tellurion/line_extractor/resolve/main/isnetis.safetensors",
    "ISNet-sketch": "https://huggingface.co/tellurion/line_extractor/resolve/main/sketch-segment.safetensors"
}


def rmbg_proceed(self, x: torch.Tensor, th=None, tw=None, dilate=False, *args, **kwargs):
    b, c, h, w = x.shape
    x = (x + 1.0) / 2.
    x = tf.resize(x, [1024, 1024])
    x = tf.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = self(x)[-1].sigmoid()
    x = tf.resize(x, [h, w])

    if th and tw:
        x = tf.pad(x, padding=[(th-h)//2, (tw-w)//2])
    if dilate:
        x = F.max_pool2d(x, kernel_size=11, stride=1, padding=5)
    x = x.clamp(0, 1)
    return x



def create_model(model="lineart"):
    """Create a model for anime2sketch
    hardcoding the options for simplicity
    """
    # if model == "LangSAM2":
    #     return LangSAM("sam2.1_hiera_large")

    if model == "rmbg-v2":
        model = AutoModelForImageSegmentation.from_pretrained(
            'briaai/RMBG-2.0',
            trust_remote_code = True,
            cache_dir = model_path,
        )
        model.eval()
        model.proceed = rmbg_proceed.__get__(model, model.__class__)
        return model

    assert model in remote_model_dict.keys()
    remote_path = remote_model_dict[model]
    basename = os.path.basename(remote_path)
    ckpt_path = os.path.join(model_path, basename)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(ckpt_path):
        cache_path = "preprocessor/weights/weights.tmp"
        download_url_to_file(remote_path, dst=cache_path)
        os.rename(cache_path, ckpt_path)

    if model == "lineart":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    elif model == "lineart_denoise":
        net = res_skip()
    elif model == "lineart_keras":
        net = SketchKeras()
    elif model == "lineart_sk":
        net = LineartDetector()
    elif model == "ISNet" or model == "ISNet-sketch":
        net = ISNetDIS()
    else:
        return None

    ckpt = load_weights(ckpt_path)
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    net.load_state_dict(ckpt)
    return net.eval()