import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils.checkpoint import checkpoint

import numpy as np
import numpy.random as random
import itertools
import importlib

from tqdm import tqdm
from inspect import isfunction
from functools import wraps
from PIL import Image, ImageDraw, ImageFont



def exists(x):
    return x is not None

def append_dims(x, target_dims) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_drop(x, p, dim=0):
    return append_dims(torch.bernoulli((1 - p) * torch.ones(x.shape[dim], device=x.device, dtype=x.dtype)), x.ndim)


def expand_to_batch_size(x, bs):
    if isinstance(x, list):
        x = [xi.repeat(bs, *([1] * (len(xi.shape) - 1))) for xi in x]
    else:
        x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
    return x


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def scaled_resize(x: torch.Tensor, scale_factor, interpolation_mode="bicubic"):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation_mode)

def get_crop_scale(h, w, bgh, bgw):
    gen_aspect = w / h
    bg_aspect = bgw / bgh
    if gen_aspect > bg_aspect:
        cw = 1.0
        ch = (h / w) * (bgw / bgh)
    else:
        ch = 1.0
        cw = (w / h) * (bgh / bgw)
    return ch, cw

def warp_resize(x: torch.Tensor, target_size, interpolation_mode="bicubic"):
    assert len(x.shape) == 4
    return F.interpolate(x, size=target_size, mode=interpolation_mode)

def resize_and_crop(x: torch.Tensor, ch, cw, th, tw):
    b, c, h, w = x.shape
    return tf.resized_crop(x, 0, 0, int(ch * h), int(cw * w), size=[th, tw])


def fitting_weights(model, sd):
    n_params = len([name for name, _ in
                    itertools.chain(model.named_parameters(),
                                    model.named_buffers())])
    for name, param in tqdm(
            itertools.chain(model.named_parameters(),
                            model.named_buffers()),
            desc="Fitting old weights to new weights",
            total=n_params
    ):
        if not name in sd:
            continue
        old_shape = sd[name].shape
        new_shape = param.shape
        assert len(old_shape) == len(new_shape)
        if len(new_shape) > 2:
            # we only modify first two axes
            assert new_shape[2:] == old_shape[2:]
        # assumes first axis corresponds to output dim
        if not new_shape == old_shape:
            new_param = param.clone()
            old_param = sd[name]
            if len(new_shape) == 1:
                # Vectorized 1D case
                new_param = old_param[torch.arange(new_shape[0]) % old_shape[0]]
            elif len(new_shape) >= 2:
                # Vectorized 2D case
                i_indices = torch.arange(new_shape[0])[:, None] % old_shape[0]
                j_indices = torch.arange(new_shape[1])[None, :] % old_shape[1]

                # Use advanced indexing to extract all values at once
                new_param = old_param[i_indices, j_indices]

                # Count how many times each old column is used
                n_used_old = torch.bincount(
                    torch.arange(new_shape[1]) % old_shape[1],
                    minlength=old_shape[1]
                )

                # Map to new shape
                n_used_new = n_used_old[torch.arange(new_shape[1]) % old_shape[1]]

                # Reshape for broadcasting
                n_used_new = n_used_new.reshape(1, new_shape[1])
                while len(n_used_new.shape) < len(new_shape):
                    n_used_new = n_used_new.unsqueeze(-1)

                # Normalize
                new_param = new_param / n_used_new

            sd[name] = new_param
    return sd


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def random_mask(img, grid_num, grid_size=128, image_size=512, mask_value=1):
    max_grid_num = image_size // grid_size
    grids = random.randint(max_grid_num, size=(grid_num,2))
    for grid in grids:
        r, c = grid
        img[:, :, r*grid_size: (r+1)*grid_size, c*grid_size: (c+1)*grid_size] = mask_value
    return img

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        lines = "\n".join(
            text_seq[start : start + nc] for start in range(0, len(text_seq), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def background_bleaching(x, xs, xc, xms, xmr, thresh_s, thresh_r, p, dtype=torch.float32):
    bs = xc.shape[0]
    thresh_s, thresh_r = append_dims(thresh_s, xms.ndim), append_dims(thresh_r, xms.ndim)
    white_bg_idx = append_dims(torch.rand((bs,), device=xc.device) < p, xc.ndim)
    white_bg = torch.ones_like(xc) * (
            torch.rand((bs, 1, 1, 1), device=xc.device, dtype=dtype) * 0.05 + 0.95
    )
    x = torch.where(
        white_bg_idx,
        torch.where(xms > thresh_s, x, torch.ones_like(x)),
        x
    )
    xs = torch.where(xms > thresh_s, xs, -torch.ones_like(xs))
    xc_bg = torch.where(
        white_bg_idx,
        white_bg,
        torch.where(xmr <= thresh_r, xc, torch.ones_like(xc))
    )
    return x, xs, xc_bg


def mask_thresholding(mask: torch.Tensor, ts):
    if isinstance(ts, torch.Tensor):
        ts = append_dims(ts, mask.ndim)
    return torch.where(mask > ts, torch.ones_like(mask), torch.zeros_like(mask))


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def checkpoint_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'checkpoint') or self.checkpoint:
            def bound_func(*args, **kwargs):
                return func(self, *args, **kwargs)
            return checkpoint(bound_func, *args, use_reentrant=False, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper
