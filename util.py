import re
import sys
import torch
import os.path as osp

from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from libs.convert_ckpt import convert_sd_ckpt

VALID_FORMATS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]


def get_format(filename):
    return osp.splitext(filename)[-1]


def load_config(path):
    # Load config file
    try:
        configs = OmegaConf.load(path)
    except:
        raise IOError("Failed in loading model configs, please check the training settings.")
    print(f"Loaded model config from {path}")
    return configs


def load_weights(path, weights_only=True):
    ext = get_format(path)
    assert ext in VALID_FORMATS, f"Invalid checkpoint format {ext}"
    if ext == ".safetensors":
        sd = {}
        safe_sd = safe_open(path, framework="pt", device="cpu")
        for key in safe_sd.keys():
            sd[key] = safe_sd.get_tensor(key)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=weights_only)
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
    return sd


def package_weights(filename, *args):
    sd = {}
    for path in args:
        print(path)
        sd.update(load_weights(path))
    save_file(sd, f"{filename}")


def delete_states(sd, delete_keys: list[str] = (), skip_keys: list[str] = ()):
    keys = list(sd.keys())
    for k in keys:
        for ik in delete_keys:
            if len(skip_keys) > 0:
                for sk in skip_keys:
                    if re.match(ik, k) is not None and re.match(sk, k) is None:
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            else:
                if re.match(ik, k) is not None:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
    return sd


def filter_ema(sd):
    new_sd = {}
    for key in sd.keys():
        if key.find("cond_stage_model") > -1 or key.find("model_ema") > -1:
            continue
        elif key.find("model.diffusion_model") > -1:
            new_sd[key] = sd[key.replace(".", "").replace("modeldiff", "model_ema.diff")].clone()
        else:
            new_sd[key] = sd[key].clone()
    return new_sd


def retrieve_ema(ckpt, filename=None):
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)
    new_sd = filter_ema(sd)
    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(new_sd, f"{filename}")


def copy_unet_weights(ckpt_path, filename=None):
    sd = load_weights(ckpt_path)
    new_sd = sd.copy()
    for key in sd.keys():
        if key.find("model.diffusion_model.semantic_input_blocks") > -1:
            continue
        if key.find("model.diffusion_model") > -1:
            nk = key.replace("model", "refnet")
            new_sd[nk] = sd[key].clone()
    filename = f"{filename}.safetensors" if filename is not None else "integrated.safetensors"
    save_file(new_sd, filename)


def postprocess_weights(ckpt_path, filename=None, exclude_frozen_params=False, ema=True):
    from libs.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    sd = convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, exclude_frozen_parameters=exclude_frozen_params)

    if ema:
        new_sd = filter_ema(sd)
    else:
        new_sd = {}
        for key in sd.keys():
            if key.find("cond_stage_model") > -1 or key.find("model_ema") > -1:
                continue
            new_sd[key] = sd[key].clone()

    filename = f"{osp.basename(ckpt_path)}.safetensors" if filename is None else filename
    save_file(new_sd, f"{filename}")


def post_with_ema(*args, **kwargs):
    postprocess_weights(ema=True, *args, **kwargs)

def post_without_ema(*args, **kwargs):
    postprocess_weights(ema=False, *args, **kwargs)


def filter_weights(ckpt, delete_keys, filename=None, skip_keys=None):
    skip_keys = [skip_keys] if skip_keys is not None else []
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)
    sd = delete_states(sd, [delete_keys], skip_keys)
    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(sd, f"{filename}")


def convert_ckpt(checkpoint, new_checkpoint):
    sd = load_weights(checkpoint)
    save_file(convert_sd_ckpt(sd), new_checkpoint)


if __name__ == '__main__':
    functions = {
        "post": post_without_ema,
        "post-ema": post_with_ema,
        "merge": package_weights,
        "filter": filter_weights,
        "ema": retrieve_ema,
        "copy": copy_unet_weights,
        "convert": convert_ckpt
    }
    args = sys.argv[1:]
    func = functions[args[0]]
    print(f"Applying util function {func.__name__}...")
    func(*args[1:])
    print(f"Process finished.")