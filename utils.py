import importlib
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

def exist(v):
    return v is not None


def default(opt, v, d=None):
    if hasattr(opt, v) and getattr(opt, v):
        return getattr(opt, v)
    return d


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def to_grayscale(data):
    grayscale = transforms.Grayscale(num_output_channels=3)(data)
    return grayscale


def ema_smooth(scalars, weight=0.9):
    last = scalars[0]
    smoothed_scalars =[]
    for point in scalars:
        smoothed_t = last * weight + (1 - weight) * point
        smoothed_scalars.append(smoothed_t)
        last = smoothed_t
    return smoothed_scalars


def get_log_loss(path, key, elen=72):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    loss_log = ea.scalars.Items(key)
    step, loss = [], []
    for t in loss_log[:elen]:
        step.append(t.step)
        loss.append(t.value)
    loss = ema_smooth(loss)
    return step, loss


def save_image(data, filename, grayscale=False):
    """
        image should be a torch.Tensor().cpu() [c, h, w]
        rgb value: [-1, 1] -> [0, 255]
    """

    img = (data.clone() + 1.) * 127.5

    img = img.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    if grayscale:
        img = img[:, :, 0]
    img = Image.fromarray(img)
    img.save(filename)


def format_time(second):
    s = int(second)
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)