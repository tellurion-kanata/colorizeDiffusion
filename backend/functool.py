import cv2
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from functools import partial


maxium_resolution = 4096
token_length = int(256 ** 0.5)

def exists(v):
    return v is not None

resize = partial(transforms.Resize, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

def resize_image(img, new_size, w, h):
    if w > h:
        img = resize((int(h / w * new_size), new_size))(img)
    else:
        img = resize((new_size, int(w / h * new_size)))(img)
    return img

def pad_image(image: torch.Tensor, h, w):
    b, c, height, width = image.shape
    square_image = -torch.ones([b, c, h, w], device=image.device)
    left = (w - width) // 2
    top = (h - height) // 2
    square_image[:, :, top:top+height, left:left+width] = image

    return square_image, (left, top, width, height)


def pad_image_with_margin(image: Image, scale):
    w, h = image.size
    nw = int(w * scale)
    bg = Image.new('RGB', (nw, h), (255, 255, 255))
    bg.paste(image, ((nw-w)//2, 0))
    return bg


def crop_image_from_square(square_image, original_dim):
    left, top, width, height = original_dim
    return square_image.crop((left, top, left + width, top + height))


def to_tensor(x, inverse=False):
    x = transforms.ToTensor()(x).unsqueeze(0)
    x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x).cuda()
    return x if not inverse else -x

def to_numpy(x, not_mask=True):
    if not_mask:
        return ((x.clamp(-1, 1) + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    else:
        return (x.clamp(0, 1) * 255)[0][0].cpu().numpy().astype(np.uint8)

def lineart_standard(x: Image.Image):
    x = np.array(x).astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    intensity = np.repeat(np.expand_dims(intensity, 2), 3, axis=2)
    result = to_tensor(intensity.clip(0, 255).astype(np.uint8))
    return result

def preprocess_sketch(sketch, resolution, preprocess="none", extractor=None):
    w, h = sketch.size
    th, tw = resolution
    r = min(th/h, tw/w)

    if preprocess == "none":
        sketch = to_tensor(sketch)
    elif preprocess == "invert":
        sketch = to_tensor(sketch, inverse=True)
    elif preprocess == "invert-webui":
        sketch = lineart_standard(sketch)
    else:
        sketch = extractor.proceed(resize((512, 512))(sketch))

    sketch, original_shape = pad_image(resize((int(h*r), int(w*r)))(sketch), th, tw)
    white_sketch = -sketch
    return sketch, original_shape, white_sketch


@torch.no_grad()
def preprocessing_inputs(
        sketch: Image.Image,
        reference: Image.Image,
        background: Image.Image,
        preprocess: str,
        hook: bool,
        resolution: tuple[int, int],
        extractor: nn.Module,
        pad_scale: float = 1.
):
    extractor = extractor.cuda()
    h, w = resolution
    if exists(sketch):
        sketch, original_shape, white_sketch = preprocess_sketch(sketch, resolution, preprocess, extractor)
    else:
        sketch = -torch.ones([1, 3, h, w], device="cuda")
        white_sketch = None
        original_shape = (0, 0, h, w)

    if hook:
        assert exists(reference) and exists(extractor)
        maxm = max(h, w)
        inject_xs = extractor.proceed(resize((maxm, maxm))(reference))
        inject_xr = to_tensor(resize((h, w))(reference))
    else:
        inject_xs = None
        inject_xr = None
    extractor = extractor.cpu()

    if exists(reference):
        if pad_scale > 1.:
            reference = pad_image_with_margin(reference, pad_scale)
        reference = to_tensor(reference)

    if exists(background):
        if pad_scale > 1.:
            background = pad_image_with_margin(background, pad_scale)
        background = to_tensor(background)

    return sketch, reference, background, original_shape, inject_xr, inject_xs, white_sketch

def postprocess(results, sketch, reference, background, crop, original_shape,
                mask_guided, smask, rmask, bgmask, mask_ts, mask_ss):
    results = to_numpy(results)
    sketch = to_numpy(sketch)[0]

    results_list = []
    for result in results:
        result = Image.fromarray(result)
        if crop:
            result = crop_image_from_square(result, original_shape)
        results_list.append(result)

    results_list.append(sketch)

    if exists(reference):
        reference = to_numpy(reference)[0]
        results_list.append(reference)
        # if vis_crossattn:
        #     results_list += visualize_attention_map(reference, results_list[0], vh, vw)

    if exists(background):
        background = to_numpy(background)[0]
        results_list.append(background)

        if exists(bgmask):
            background = Image.fromarray(background)
            results_list.append(Image.composite(
                background,
                Image.new("RGB", background.size, (255, 255, 255)),
                Image.fromarray(to_numpy(bgmask, not_mask=False), mode="L")
            ))
            results_list.append(Image.composite(
                Image.new("RGB", background.size, (255, 255, 255)),
                background,
                Image.fromarray(to_numpy(bgmask, not_mask=False), mode="L")
            ))

    if mask_guided:
        smask[smask < mask_ss] = 0
        results_list.append(Image.fromarray(to_numpy(smask, not_mask=False), mode="L"))

        if exists(rmask):
            reference = Image.fromarray(reference)
            rmask[rmask < mask_ts] = 0
            results_list.append(Image.fromarray(to_numpy(rmask, not_mask=False), mode="L"))
            results_list.append(Image.composite(
                reference,
                Image.new("RGB", reference.size, (255, 255, 255)),
                Image.fromarray(to_numpy(rmask, not_mask=False), mode="L")
            ))
            results_list.append(Image.composite(
                Image.new("RGB", reference.size, (255, 255, 255)),
                reference,
                Image.fromarray(to_numpy(rmask, not_mask=False), mode="L")
            ))

    return results_list


def parse_prompts(
        prompts: str,
        target: bool = None,
        anchor: bool = None,
        control: bool = None,
        target_scale: bool = None,
        ts0: float = None,
        ts1: float = None,
        ts2: float = None,
        ts3: float = None,
        enhance: bool = None
):

    targets = []
    anchors = []
    controls = []
    scales = []
    enhances = []
    thresholds_list = []

    replace_str = ["; [anchor] ", "; [control] ", "; [scale]", "; [enhanced]", "; [ts0]", "; [ts1]", "; [ts2]", "; [ts3]"]
    if prompts != "" and prompts is not None:
        ps_l = prompts.split('\n')
        for ps in ps_l:
            ps = ps.replace("[target] ", "")
            for str in replace_str:
                ps = ps.replace(str, "||||")

            p_l = ps.split("||||")
            targets.append(p_l[0])
            anchors.append(p_l[1])
            controls.append(p_l[2])
            scales.append(float(p_l[3]))
            enhances.append(bool(p_l[4]))
            thresholds_list.append([float(p_l[5]), float(p_l[6]), float(p_l[7]), float(p_l[8])])

    if exists(target) and target != "":
        targets.append(target)
        anchors.append(anchor)
        controls.append(control)
        scales.append(target_scale)
        enhances.append(enhance)
        thresholds_list.append([ts0, ts1, ts2, ts3])

    return {
        "targets": targets,
        "anchors": anchors,
        "controls": controls,
        "target_scales": scales,
        "enhances": enhances,
        "thresholds_list": thresholds_list
    }


from refnet.sampling.manipulation import get_heatmaps
def visualize_heatmaps(model, reference, manipulation_params, control, ts0, ts1, ts2, ts3):
    if reference is None:
        return []

    size = reference.size
    if size[0] > maxium_resolution or size[1] > maxium_resolution:
        if size[0] > size[1]:
            size = (maxium_resolution, int(float(maxium_resolution) / size[0] * size[1]))
        else:
            size = (int(float(maxium_resolution) / size[1] * size[0]), maxium_resolution)
        reference = reference.resize(size, Image.BICUBIC)

    reference = np.array(reference)
    scale_maps = get_heatmaps(model, to_tensor(reference), size[1], size[0],
                              control, ts0, ts1, ts2, ts3, **manipulation_params)

    scale_map = scale_maps[0] + scale_maps[1] + scale_maps[2] + scale_maps[3]
    heatmap = cv2.cvtColor(cv2.applyColorMap(scale_map, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(reference, 0.3, heatmap, 0.7, 0)
    hu = size[1] // token_length
    wu = size[0] // token_length
    for i in range(16):
        result[i * hu, :] = (0, 0, 0)
    for i in range(16):
        result[:, i * wu] = (0, 0, 0)

    return [result]