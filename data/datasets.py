import json
import os

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import pytorch_lightning as pl

import numpy.random as random
import PIL.Image as Image

from os.path import *
from glob import glob
from PIL import ImageFile
from collections import namedtuple


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
ImageFile.LOAD_TRUNCATED_IMAGES = True

# OpenAI mean & std
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_LOAD_SIZE = 224


def exists(v):
    return v is not None

def get_transform_seeds(t, load_size=256, crop_size=None, rotate_p=0.2):
    seed_range = t.rotate_range
    seeds = [random.randint(-seed_range, seed_range),
             random.randint(-seed_range, int(seed_range*rotate_p))]
    if crop_size == load_size:
        return seeds, None
    top, left = random.randint(0, load_size - crop_size, 2)
    crops = [top, left, crop_size]
    return seeds, crops


def custom_transform(img, seeds, t, load_size, crops=None):
    range_seed, rotate_flag = seeds[:]
    if t.flip and range_seed > 0:
        img = tf.hflip(img)
    if t.rotate and rotate_flag > 0:
        img = tf.rotate(img, range_seed, fill=255)
    if t.resize:
        if exists(crops):
            img = tf.resize(img, load_size)
        else:
            img = tf.resize(img, [load_size, load_size])
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


def normalize(img,
              grayscale=False,
              mean=(0.5, 0.5, 0.5),
              std=(0.5, 0.5, 0.5)):
    img = transforms.ToTensor()(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize(mean, std)(img)
    return img


def resize_with_ratio(img, new_size):
    """ This function resizes the longer edge to new_size, instead of the shorter one in PyTorch """
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size))(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)))(img)
    return img


class ImageDataset(data.Dataset):
    def __init__(self,
                 dataroot,
                 mode,
                 eval_load_size=None,
                 load_size=256,
                 crop_size=None,
                 transform_list={},
                 keep_ratio=False):
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']
        self.load_size = load_size
        self.crop_size = crop_size

        self.image_files = [file for ext in IMAGE_EXTENSIONS
                            for file in glob(join(dataroot, '*.{}'.format(ext)))]
        self.image_files += [file for ext in IMAGE_EXTENSIONS
                             for file in glob(join(dataroot, '*/*.{}'.format(ext)))]
        if not self.eval:
            named_transforms = namedtuple('Transforms', transform_list.keys())
            self.transforms = named_transforms(**transform_list)
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
        else:
            self.eval_load_size = eval_load_size
            self.kr = keep_ratio

    def __getitem__(self, index):
        image_file = self.image_files[index]
        img = Image.open(image_file).convert('RGB')
        filename = basename(image_file)
        img_idx = splitext(filename)[0]

        if not self.eval:
            # flip, crop and resize in custom transform function
            seeds, crops = get_transform_seeds(self.transforms, self.load_size, self.crop_size)
            img = custom_transform(img, seeds, self.transforms, self.load_size, crops)
        else:
            if self.kr and self.eval_load_size:
                img = resize_with_ratio(img, self.eval_load_size)
            elif self.eval_load_size:
                img = transforms.Resize((self.eval_load_size, self.eval_load_size))(img)
        img = normalize(img)
        return {"color": img, "index": img_idx}

    def __len__(self):
        return len(self.image_files)


class ReferenceDataset(ImageDataset):
    '''
        Dataloader for reference feature vector-based generation
        Using the same image as reference input.
    '''

    def __init__(self,
                 use_tokens=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        ref_load_size = CLIP_LOAD_SIZE
        if self.eval and use_tokens:
            ref_load_size = int(self.load_size / self.crop_size * CLIP_LOAD_SIZE)
            assert (ref_load_size / CLIP_LOAD_SIZE) == (self.load_size / self.crop_size)
        self.ref_load_size, self.ref_crop_size = ref_load_size, ref_load_size
        self.ref_mean, self.ref_std = OPENAI_MEAN, OPENAI_STD

    def __getitem__(self, index):
        image_file = self.image_files[index]
        img = Image.open(image_file).convert('RGB')
        filename = basename(image_file)
        img_idx = splitext(filename)[0]

        if not self.eval:
            # flip, crop and resize in custom transform function
            seeds, crops = get_transform_seeds(self.transforms, self.load_size, self.crop_size)
            img = custom_transform(img, seeds, self.transforms, self.load_size, crops)
            ref = transforms.Resize(self.ref_load_size)(img)
        else:
            ref = img
            if self.kr and self.eval_load_size:
                img = resize_with_ratio(img, self.eval_load_size)
            elif self.eval_load_size:
                img = transforms.Resize((self.eval_load_size, self.eval_load_size))(img)
            ref = transforms.Resize((self.ref_load_size, self.ref_load_size))(ref)

        img = normalize(img)
        ref = normalize(ref, mean=self.ref_mean, std=self.ref_std)
        return {
            "color": img,
            "reference": ref,
            "index": img_idx
        }


class DraftDataset(data.Dataset):
    # TODO: remove use_clip option
    def __init__(self,
                 dataroot,
                 mode,
                 eval_load_size=None,
                 save_input=True,
                 refset_key="reference",
                 load_size=256,
                 crop_size=256,
                 transform_list={},
                 keep_ratio=False,
                 ref_load_size=CLIP_LOAD_SIZE,
                 use_tokens=False,
                 ):
        assert mode in ['train', 'test', 'validation'], f'Dataset mode {mode} does not exist.'
        self.eval = mode in ['test', 'validation']
        self.load_size = load_size
        self.crop_size = crop_size

        dataroot = abspath(dataroot)
        self.sketch_root = join(dataroot, 'sketch')
        self.color_root = join(dataroot, 'color') if save_input else self.sketch_root
        self.ref_root = join(dataroot, refset_key) if not self.eval else join(dataroot, 'reference') # use reference image in validation/testing
        self.image_files = [file for ext in IMAGE_EXTENSIONS
                            for file in glob(join(self.sketch_root, '*.{}'.format(ext)))]
        self.image_files += [file for ext in IMAGE_EXTENSIONS
                             for file in glob(join(self.sketch_root, '*/*.{}'.format(ext)))]

        self.data_size = len(self)
        self.offset = random.randint(0, self.data_size) if mode == 'validation' else 0
        self.use_clip = exists(ref_load_size)

        if not self.eval:
            self.preprocess = self.training_preprocess

            named_transforms = namedtuple('Transforms', transform_list.keys())
            self.transforms = named_transforms(**transform_list)
            self.crop_size = crop_size
            assert load_size >= crop_size, f'load size {load_size} should not be smaller than crop size {crop_size}'
        else:
            self.preprocess = self.testing_preprocess

            self.load_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio
            if self.use_clip:
                # crop_size is the training image size for the model
                new_load_size = int(self.load_size / self.crop_size * ref_load_size)
                assert (new_load_size / ref_load_size) == (self.load_size / self.crop_size)
                ref_load_size = new_load_size

        if self.use_clip:
            self.ref_load_size, self.ref_crop_size = (ref_load_size, ref_load_size) if use_tokens else (CLIP_LOAD_SIZE, CLIP_LOAD_SIZE)
            self.ref_mean, self.ref_std = OPENAI_MEAN, OPENAI_STD
        else:
            self.ref_load_size, self.ref_crop_size = self.load_size, self.crop_size
            self.ref_mean, self.ref_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def get_images(self, index):
        filename = self.image_files[index]
        img_idx = splitext(basename(filename))[0]

        ske = Image.open(filename).convert('L')
        col = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')

        if self.offset > 0:
            ref_file = self.image_files[(index + self.offset) % self.data_size]
            filename = basename(ref_file)
            ref = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.sketch_root, self.ref_root)).convert('RGB')
        return [ske, col, ref], img_idx

    def training_preprocess(self, ske, col, ref):
        # flip, crop and resize in custom transform function
        seeds, crops = get_transform_seeds(self.transforms, self.load_size, self.crop_size)
        ske, col = map(lambda t:
                            custom_transform(t, seeds, self.transforms, self.load_size, crops),
                            (ske, col))

        # CLIP image encoder takes different input resolution
        seeds, crops = get_transform_seeds(self.transforms, self.ref_load_size, self.ref_crop_size)
        ref = custom_transform(ref, seeds, self.transforms, self.ref_load_size, crops)

        return ske, col, ref

    def testing_preprocess(self, ske, col, ref):
        if self.kr:
            ske, col = map(lambda t:
                                resize_with_ratio(t, self.load_size),
                                (ske, col))
        else:
            ske, col = map(lambda t:
                                transforms.Resize((self.load_size, self.load_size))(t),
                                (ske, col))
        ref = transforms.Resize((self.load_size, self.load_size))(ref) \
            if not self.use_clip else transforms.Resize((self.ref_load_size, self.ref_load_size))(ref)
        return ske, col, ref

    def normalize_inputs(self, imgs):
        ske, col, ref = imgs[:]
        ske, col , ref = self.preprocess(ske, col, ref)

        ske = normalize(ske, grayscale=True)
        col = normalize(col)
        ref = normalize(ref, mean=self.ref_mean, std=self.ref_std)
        return ske, col, ref

    def __getitem__(self, index):
        imgs, img_idx = self.get_images(index)
        ske, col, ref = self.normalize_inputs(imgs)

        return {
            "sketch": ske,
            "color": col,
            "reference": ref,
            "index": img_idx
        }

    def __len__(self):
        return len(self.image_files)


class TextDataset(DraftDataset):
    def __init__(self, dataroot, **kwargs):
        super().__init__(dataroot, **kwargs)
        self.tag_root = join(abspath(dataroot), "tags")
        self.image_files = glob(join(self.tag_root, '*.json'))
        self.image_files += glob(join(self.tag_root, '*/*.json'))

    def get_images(self, index):
        filename = self.image_files[index]
        img_idx = splitext(basename(filename))[0]
        with open(filename, 'r') as file:
            dict = json.load(file)
            text = dict["tag_string"]
            # tags = dict[""]
            filename = dict["linked_img"]

        ske = Image.open(filename).convert('L')
        col = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')

        if self.offset > 0:
            ref_file = self.image_files[(index + self.offset) % self.data_size]
            filename = basename(ref_file.replace(self.sketch_root, self.ref_root))
            ref = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.sketch_root, self.ref_root)).convert('RGB')
        return [ske, col, ref], img_idx, text

    def __getitem__(self, index):
        imgs, img_idx, text = self.get_images(index)
        ske, col, ref = self.normalize_inputs(imgs)

        return {
            "sketch": ske,
            "color": col,
            "reference": ref,
            "text": text,
            "index": img_idx
        }


class CustomDataLoader(pl.LightningDataModule):
    def __init__(self, opt, cfg, device_num, eval_load_size=None, save_input=True):
        super().__init__()
        DATALOADER = {
            'ImageLoader': ImageDataset,
            'ReferLoader': ReferenceDataset,
            'DraftLoader': DraftDataset,
            'TextLoader': TextDataset
        }

        loader_cls = cfg['class'] if not opt.eval else cfg.get("eval_class", cfg['class'])
        assert loader_cls in DATALOADER.keys(), f'DataLoader {loader_cls} does not exist.'
        loader = DATALOADER[loader_cls]

        self.dataset = loader(
            dataroot        = opt.dataroot,
            mode            = opt.mode,
            eval_load_size  = eval_load_size,
            save_input      = save_input,
            **cfg['params']
        )

        self.dataLoader = data.DataLoader(
            dataset     = self.dataset,
            batch_size  = opt.batch_size,
            shuffle     = cfg.get("shuffle", True) and not opt.eval,
            num_workers = opt.num_threads,
            drop_last   = device_num > 1,
        )

    def train_dataloader(self):
        return self.dataLoader

    def test_dataloader(self):
        return self.dataLoader

    def get_data_size(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataLoader:
            yield data

    def __len__(self):
        return len(self.dataLoader)