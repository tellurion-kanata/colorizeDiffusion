import os.path as osp
import numpy.random as random
import PIL.Image as Image

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from glob import glob
from PIL import ImageFile
from collections import namedtuple
from functools import partial


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
ImageFile.LOAD_TRUNCATED_IMAGES = True
bic = transforms.InterpolationMode.BICUBIC

def exists(v):
    return v is not None


def get_transform_seeds(t, load_size=512, crop_size=512, rotate_p=0.2):
    seed_range = t.rotate_range
    seeds = [random.randint(-seed_range, seed_range),
             random.randint(-seed_range, int(seed_range*rotate_p))]

    if crop_size == load_size:
        crops = None
    else:
        top, left = random.randint(0, load_size - crop_size, 2)
        crops = [top, left, crop_size]
    return seeds, crops


def custom_transform(img, seeds, crops, t, load_size=512):
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
        top, left, length = crops[:]
        img = tf.crop(img, top, left, length, length)
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


class RefDataset(data.Dataset):
    def __init__(
            self,
            dataroot,
            mode = "train",
            eval_load_size = None,
            refset_key = "reference",
            load_size = 256,
            crop_size = 256,
            transform_list = {},
            keep_ratio = False,
        ):
        super().__init__()
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']

        dataroot = osp.abspath(dataroot)
        self.color_root = osp.join(dataroot, 'color')
        self.ref_root = osp.join(dataroot, refset_key)
        self.image_files = [
            file for ext in IMAGE_EXTENSIONS
            for file in glob(osp.join(self.color_root, f'*.{ext}'))+glob(osp.join(self.color_root, f'*/*.{ext}'))
        ]

        self.data_size = len(self)
        self.offset = random.randint(0, self.data_size) if mode == 'validation' else 0

        if not self.eval:
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
            self.preprocess = self.training_preprocess

            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)
            self.get_seeds = partial(get_transform_seeds, transforms, load_size, crop_size)
            self.custom_transforms = partial(custom_transform, t=transforms, load_size=load_size)
        else:
            self.preprocess = self.testing_preprocess
            self.load_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio

    def get_images(self, index):
        filename = self.image_files[index]
        col = Image.open(filename).convert('RGB')

        if self.offset > 0:
            ref_file = self.image_files[(index + self.offset) % self.data_size]
            ref = Image.open(ref_file.replace(self.color_root, self.ref_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.color_root, self.ref_root)).convert('RGB')
        return self.preprocess(col, ref)

    def training_preprocess(self, col, ref):
        # flip, crop and resize in custom transform function
        seeds, crops = self.get_seeds()
        col, ref = map(lambda t: self.custom_transforms(t, seeds, crops), (col, ref))
        return col, ref

    def testing_preprocess(self, col, ref):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        col, ref = map(lambda t: resize(t, self.load_size), (col, ref))
        return col, ref

    def __getitem__(self, index):
        col, ref = self.get_images(index)
        col, ref = map(lambda t: normalize(t), (col, ref))

        return {
            "image": col,
            "reference": ref,
        }

    def __len__(self):
        return len(self.image_files)


class SketchDataset(data.Dataset):
    def __init__(
            self,
            dataroot,
            mode = "train",
            eval_load_size = None,
            load_size = 256,
            crop_size = 256,
            transform_list = {},
            keep_ratio = False,
            inverse_grayscale = False,
        ):
        super().__init__()
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']

        dataroot = osp.abspath(dataroot)
        self.sketch_root = osp.join(dataroot, 'sketch')
        self.color_root = osp.join(dataroot, 'color')
        self.image_files = [
            file for ext in IMAGE_EXTENSIONS
            for file in glob(osp.join(self.sketch_root, f'*.{ext}'))+glob(osp.join(self.sketch_root, f'*/*.{ext}'))
        ]

        self.inverse_grayscale = inverse_grayscale
        self.data_size = len(self)
        self.offset = random.randint(0, self.data_size) if mode == 'validation' else 0

        if not self.eval:
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
            self.preprocess = self.training_preprocess

            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)
            self.get_transform_seeds = partial(get_transform_seeds, transforms, load_size, crop_size)
            self.custom_transform = partial(custom_transform, t=transforms, load_size=load_size)
        else:
            self.preprocess = self.testing_preprocess
            self.load_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio

    def get_images(self, index):
        filename = self.image_files[index]

        ske = Image.open(filename).convert('RGB')
        col = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')
        return self.preprocess(ske, col)

    def training_preprocess(self, ske, col):
        # flip, crop and resize in custom transform function
        seeds, crops = self.get_transform_seeds()
        ske, col = map(lambda t:self.custom_transform(t, seeds, crops), (ske, col))
        return ske, col

    def testing_preprocess(self, ske, col):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        ske, col = map(lambda t: resize(t, self.load_size), (ske, col))
        return ske, col

    def __getitem__(self, index):
        ske, col = self.get_images(index)
        ske = 1 - to_tensor(ske) if self.inverse_grayscale else to_tensor(ske)
        return {
            "sketch": ske,
            "image": normalize(col),
        }

    def __len__(self):
        return len(self.image_files)


class TripletDataset(data.Dataset):
    def __init__(
            self,
            dataroot,
            mode = "train",
            eval_load_size = None,
            refset_key = "reference",
            load_size = 512,
            crop_size = 512,
            transform_list = {},
            keep_ratio = False,
            inverse_grayscale = False,
        ):
        super().__init__()
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']

        dataroot = osp.abspath(dataroot)
        self.sketch_root = osp.join(dataroot, 'sketch')
        self.color_root = osp.join(dataroot, 'color')
        self.ref_root = osp.join(dataroot, refset_key)
        self.image_files = [
            file for ext in IMAGE_EXTENSIONS
            for file in glob(osp.join(self.sketch_root, f'*.{ext}'))+glob(osp.join(self.sketch_root, f'*/*.{ext}'))
        ]

        self.inverse_grayscale = inverse_grayscale
        self.data_size = len(self)
        self.offset = random.randint(1, self.data_size) if mode == 'validation' else 0

        if not self.eval:
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
            self.preprocess = self.training_preprocess
            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)

            self.gt_seeds = partial(get_transform_seeds, transforms, load_size, crop_size)
            self.gt_transforms = partial(custom_transform, t=transforms, load_size=load_size)

            # always use full reference image to strengthen the retrieval training
            self.ref_seeds = partial(get_transform_seeds, transforms, crop_size, crop_size)
            self.ref_transforms = partial(custom_transform, t=transforms, load_size=crop_size)
        else:
            self.preprocess = self.testing_preprocess
            self.load_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio


    def get_images(self, index):
        filename = self.image_files[index]

        ske = Image.open(filename).convert('RGB')
        col = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')

        if self.offset > 0:
            ref_file = self.image_files[(index + self.offset) % self.data_size]
            ref = Image.open(ref_file.replace(self.sketch_root, self.color_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.sketch_root, self.ref_root)).convert('RGB')
        return self.preprocess(ske, col, ref)


    def training_preprocess(self, ske, col, ref):
        # flip, crop and resize in custom transform function
        seeds, crops = self.gt_seeds()
        ske, col = map(lambda t:self.gt_transforms(t, seeds, crops), (ske, col))

        # CLIP image encoder takes different input resolution
        seeds, crops = self.ref_seeds()
        ref = self.ref_transforms(ref, seeds, crops)
        return ske, col, ref


    def testing_preprocess(self, ske, col, ref):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        ske, col, ref = map(lambda t: resize(t, self.load_size), (ske, col, ref))
        return ske, col, ref


    def __getitem__(self, index):
        ske, col, ref = map(lambda t: normalize(t), self.get_images(index))
        return {
            "control": -ske if self.inverse_grayscale else ske,
            "image": col,
            "reference": ref,
        }

    def __len__(self):
        return len(self.image_files)

import json
class TripletTextDataset(data.Dataset):
    def __init__(
            self,
            dataroot,
            mode = "train",
            eval_load_size = None,
            load_size = 512,
            crop_size = 512,
            transform_list = {},
            keep_ratio = False,
            inverse_grayscale = False,
        ):
        super().__init__()
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']

        dataroot = osp.abspath(dataroot)
        self.tag_root = osp.join(dataroot, 'tag')
        self.color_root = osp.join(dataroot, 'color')
        self.sketch_root = osp.join(dataroot, 'sketch')
        self.image_files = glob(osp.join(self.tag_root, '*/*.json'))

        self.inverse_grayscale = inverse_grayscale
        self.data_size = len(self)
        self.offset = random.randint(0, self.data_size) if mode == 'validation' else 0

        if not self.eval:
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
            self.preprocess = self.training_preprocess
            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)

            self.gt_seeds = partial(get_transform_seeds, transforms, load_size, crop_size)
            self.gt_transforms = partial(custom_transform, t=transforms, load_size=load_size)

        else:
            self.preprocess = self.testing_preprocess
            self.load_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio


    def get_inputs(self, index):
        filename = self.image_files[index]
        with open(filename, "r") as f:
            img_json = json.load(f)
        filename = img_json["linked_img"]

        txt = img_json["tag_string"] + ",anime-style"
        col = Image.open(filename).convert('RGB')
        ske = Image.open(filename.replace(self.color_root, self.sketch_root)).convert('RGB')
        ske, col = self.preprocess(ske, col)
        return txt, ske, col


    def training_preprocess(self, ske, col):
        # flip, crop and resize in custom transform function
        seeds, crops = self.gt_seeds()
        ske, col = map(lambda t:self.gt_transforms(t, seeds, crops), (ske, col))
        return ske, col


    def testing_preprocess(self, ske, col):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        ske, col = map(lambda t: resize(t, self.load_size), (ske, col))
        return ske, col


    def __getitem__(self, index):
        txt, ske, col = self.get_inputs(index)
        ske, col = map(lambda t: normalize(t), (ske, col))
        return {
            "control": -ske if self.inverse_grayscale else ske,
            "image": col,
            "txt": txt,
        }

    def __len__(self):
        return len(self.image_files)


def create_dataloader(opt, cfg, device_num, eval_load_size=None):
    DATALOADER = {
        'RefLoader': RefDataset,
        'SketchLoader': SketchDataset,
        'TripletLoader': TripletDataset,
        'TextLoader': TripletTextDataset,
    }

    loader_cls = cfg['class'] if not opt.eval else cfg.get("eval_class", cfg['class'])
    assert loader_cls in DATALOADER.keys(), f'DataLoader {loader_cls} does not exist.'
    loader = DATALOADER[loader_cls]

    dataset = loader(
        mode            = opt.mode,
        dataroot        = opt.dataroot,
        eval_load_size  = eval_load_size,
        **cfg['params']
    )

    dataloader = data.DataLoader(
        dataset     = dataset,
        batch_size  = opt.batch_size,
        shuffle     = cfg.get("shuffle", True) and not opt.eval,
        num_workers = opt.num_threads,
        drop_last   = device_num > 1,
    )
    return dataloader, len(dataset)