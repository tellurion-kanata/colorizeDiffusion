import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms

maxium_resolution = 4096
token_length = int(256 ** 0.5)


def exists(v):
    return v is not None

def resize_image(img, new_size):
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size))(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)))(img)
    return img

def pad_image_to_square(image: Image, original_shape=False, not_margin=False):
    width, height = image.size
    max_dim = max(width, height)
    padding = (0, 0, 0) if not_margin else (255, 255, 255)
    square_image = Image.new('RGB', (max_dim, max_dim), padding)
    left = (max_dim - width) / 2
    top = (max_dim - height) / 2
    square_image.paste(image, (int(left), int(top)))

    if original_shape:
        return square_image, (int(left), int(top), width, height)
    else:
        return square_image

def crop_image_from_square(square_image, original_dim):
    left, top, width, height = original_dim
    return square_image.crop((left, top, left + width, top + height))


def to_tensor(x, inverse=False):
    x = transforms.ToTensor()(x).unsqueeze(0)
    x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x).cuda()
    return x if not inverse else -x

def to_numpy(x):
    return ((x.clamp(-1, 1) + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

def preprocess_sketch(sketch, resolution, preprocess="none", extractor=None):
    sketch, original_shape = pad_image_to_square(
        resize_image(sketch, resolution),
        original_shape = True,
        not_margin = preprocess=="none",
    )
    if preprocess == "none":
        sketch = to_tensor(sketch)
    elif preprocess == "invert":
        sketch = to_tensor(sketch, inverse=True)
    else:
        sketch = extractor.proceed(sketch)
    return sketch, original_shape

@torch.no_grad()
def preprocessing_inputs(sketch, reference, preprocess, hook, resolution, extractor, use_rx):
    extractor = extractor.cuda()
    if exists(sketch):
        sketch, original_shape = preprocess_sketch(sketch, resolution, preprocess, extractor)
    else:
        sketch = torch.zeros(1, 3, resolution, resolution).cuda()
        original_shape = (0, 0, resolution, resolution)
    if hook:
        assert exists(reference) and exists(extractor)
        inject_xr = transforms.Resize((resolution, resolution))(reference)
        inject_sketch = extractor.proceed(inject_xr)
        inject_xr = to_tensor(inject_xr)
    elif use_rx:
        inject_xr = transforms.Resize((resolution, resolution))(reference)
        inject_xr = to_tensor(inject_xr)
        inject_sketch = None
    else:
        inject_xr = None
        inject_sketch = None
    extractor = extractor.cpu()
    if reference is not None:
        reference = to_tensor(reference)
    return sketch, reference, original_shape, inject_sketch, inject_xr

def parse_prompts(
        prompts : str,
        target=None,
        anchor=None,
        control=None,
        target_scale=None,
        ts0=None,
        ts1=None,
        ts2=None,
        ts3=None,
        enhance=None
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

    if exists(target):
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