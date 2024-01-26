import re
import sys
import torch
import os.path as osp

from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

VALID_FORMATS = [".pt", ".pth", ".ckpt", ".safetensors"]


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


def load_weights(path):
    ext = get_format(path)
    assert ext in VALID_FORMATS, f"Invalid checkpoint format {ext}"
    if ext == ".safetensors":
        sd = {}
        safe_sd = safe_open(path, framework="pt", device="cpu")
        for key in safe_sd.keys():
            sd[key] = safe_sd.get_tensor(key)
    else:
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
    return sd


def package_weights(save_path="./", *args):
    sd = {}
    for path in args:
        print(path)
        sd.update(load_weights(path))
    save_file(sd, osp.join(save_path, "packaged_weights.safetensors"))


def delete_states(sd, delete_keys=list(), skip_keys=list()):
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


def retrieve_ema(ckpt, filename=None):
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)

    new_sd = {}
    for key in sd.keys():
        if key.find("cond_stage_model") > -1:
            continue
        if key.find("model_ema"):
            new_sd[key.replace("model_ema.", "")] = sd[key].clone()
        if key.find("first_stage_model") > -1:
            new_sd[key] = sd[key].clone()

    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(new_sd, f"{filename}")


def postprocess_ds_weights(unet_ckpt, vae_ckpt, filename=None):
    """
        This function designed for unet trained using deepspeed mixed precision, whose weights are saved in fp16.
        Merge fp32 VAE weights into the unet ckpt.
    """
    fmt = get_format(unet_ckpt)
    assert fmt in VALID_FORMATS, "please input a pytorch checkpoint file"

    sd = load_weights(unet_ckpt)
    vae_sd = load_weights(vae_ckpt)

    new_sd = {}
    for key in sd.keys():
        if key.find("cond_stage_model") > -1:
            continue
        if key.find("model_ema"):
            new_sd[key.replace("model_ema.", "")] = sd[key].clone()

    # merge fp32 vae weights into checkpoint
    vae_keys = []
    for key in vae_sd.keys():
        if key.find("first_stage_model") > -1:
            vae_keys.append(key)
            
    new_sd.update({f"first_stage_model.{key}": vae_sd[key].clone() for key in vae_keys})
    filename = osp.basename(unet_ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    del sd, vae_sd
    save_file(new_sd, f"{filename}")

def filter_weights(ckpt, delete_keys, filename=None, skip_keys=None):
    skip_keys = [skip_keys] if skip_keys is not None else []
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)
    sd = delete_states(sd, [delete_keys], skip_keys)
    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(sd, f"{filename}")


if __name__ == '__main__':
    functions = {
        "post": postprocess_ds_weights,
        "merge": package_weights,
        "filter": filter_weights,
        "ema": retrieve_ema,
    }
    args = sys.argv[1:]
    func = functions[args[0]]
    print(f"Applying util function {func.__name__}...")
    if func == "merge":
        package_weights(args[-1], *args[1:])
    else:
        func(*args[1:])
    print(f"Process finished.")