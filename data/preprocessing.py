import PIL.Image as Image
import numpy.random as random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from typing import Union

bic = transforms.InterpolationMode.BICUBIC


def exists(v):
    return v is not None

def resize_and_pad(image, target_size=1024, final_size=1024):
    bg = Image.new('RGB', (target_size, target_size), (255,255,255))
    image = resize_with_ratio(image, target_size)
    x, y = map(lambda t: (target_size - image.size[t]) // 2, (0, 1))

    bg.paste(image, (x, y))
    resized = bg.resize((final_size, final_size), Image.BICUBIC)
    return resized

def get_transform_seeds(
        t,
        load_size: int = 512,
        crop_size: Union[int|list[int]] = 512,
        rotate_p: float = 0.2
):
    seed_range = t.rotate_range
    seeds = [random.randint(-seed_range, seed_range),
             random.randint(-seed_range, int(seed_range*rotate_p))]

    if not exists(crop_size):
        crops = None
    else:
        if not isinstance(crop_size, int):
            crop_size_h, crop_size_w = crop_size[random.randint(len(crop_size))]
            load_size = random.randint(max(crop_size_h, crop_size_w), load_size)
            top = random.randint(0, load_size - crop_size_h + 1)
            left = random.randint(0, load_size - crop_size_w + 1)
            crops = [top, left, crop_size_h, crop_size_w]

        else:
            load_size = random.randint(crop_size, load_size)
            top, left = random.randint(0, load_size - crop_size + 1, 2)
            crops = [top, left, crop_size, crop_size]
    return seeds, crops, load_size


def custom_transform(img, seeds, crops, load_size, t, center_crop_max=0):
    range_seed, rotate_flag = seeds[:]
    if t.flip and range_seed > 0:
        img = tf.hflip(img)
    if t.rotate and rotate_flag > 0:
        img = tf.rotate(img, range_seed, fill=[255,255,255])
    if t.resize:
        if exists(crops):
            img = tf.resize(img, [load_size,], bic)
        else:
            img = tf.resize(img, [load_size, load_size], bic)
    if exists(crops):
        top, left, h, w = crops[:]
        img = tf.crop(img, top, left, h, w)

        if center_crop_max > 0:
            center = random.randint(0, center_crop_max + 1)
            dir = random.randint(4)
            if dir == 0:
                img = tf.crop(img, center, 0, h-center*2, w)
                img = tf.pad(img, [0, center], fill=255)
            elif dir == 1:
                img = tf.crop(img, 0, center, h, w-center*2)
                img = tf.pad(img, [center, 0], fill=255)
            elif dir == 2:
                center2 = random.randint(0, 101)
                img = tf.crop(img, center, center2, h - center * 2, w - center2 * 2)
                img = tf.pad(img, [center2, center], fill=255)
            else:
                pass

    if t.jitter:
        seed = random.random(3) * 0.2 + 0.9
        img = jitter(img, seed)
    return img


def jitter(img, seeds):
    brt, crt, sat = seeds[:]
    img = tf.adjust_brightness(img, brt)
    img = tf.adjust_contrast(img, crt)
    img = tf.adjust_saturation(img, sat)
    return img

def to_tensor(x):
    return transforms.ToTensor()(x)

def normalize(img, grayscale=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    img = to_tensor(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize(mean, std)(img)
    return img


def resize_with_ratio(img, new_size):
    """ This function resizes the longer edge to new_size, instead of the shorter one in PyTorch """
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size), bic)(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)), bic)(img)
    return img


def resize_without_ratio(img, new_size):
    return transforms.Resize((new_size, new_size), bic)(img)


def random_erase(s: torch.Tensor, min_num=9, max_num=18, grid_size=128, image_size=512):
    max_grid_num = image_size // grid_size
    num = random.randint(min_num, max_num)
    grid_id = random.randint(max_grid_num, size=(num,2))
    for id in grid_id:
        s = tf.erase(s, id[0] * grid_size, id[1] * grid_size, grid_size, grid_size, 1, inplace=True)
    return s


def check_json(d, score_threshold, minm_resolution):
    if (d["aesthetic_score"] < score_threshold or
            d["resolution"] < minm_resolution
            or not d["exist_sketch"]):
        return False
    return True


def compute_output_padding(original_size, kernel_size, stride, downsampled_size):
    return original_size - ((downsampled_size - 1) * stride + kernel_size)


def mask_expansion(mask: torch.Tensor, grid_h, grid_w):
    if len(mask.shape) == 3:
        reshape = True
        _, h, w = mask.shape
        mask = mask.unsqueeze(0)
    else:
        reshape = False
        f, _, h, w = mask.shape

    ones = torch.ones([1, 1, grid_h, grid_w]).to(mask.device)
    mask = F.max_pool2d(mask, kernel_size=(grid_h, grid_w), stride=(grid_h, grid_w))

    output_pad_h = compute_output_padding(h, grid_h, grid_h, mask.shape[2])
    output_pad_w = compute_output_padding(w, grid_w, grid_w, mask.shape[3])
    output_padding = (output_pad_h, output_pad_w)
    expanded_mask = F.conv_transpose2d(mask, weight=ones, stride=(grid_h, grid_w), output_padding=output_padding)

    if reshape:
        expanded_mask = expanded_mask.squeeze(0)
    return expanded_mask