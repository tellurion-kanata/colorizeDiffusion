import json
import random
import zipfile
import os.path as osp
import warnings

from glob import glob

import torch
from PIL import ImageFile
from collections import namedtuple
from functools import partial
from .preprocessing import *

import torch.utils.data as data


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)                 # Too many RGBA warnings


# for ai-generated data, we set aesthetic score to 6
DEFAULT_AESTHETIC_SCORE = 6


class TripletDataset(data.Dataset):
    def __init__(
            self,
            dataroot = None,
            mode = "train",
            image_key = "color",
            control_key = "sketch",
            condition_key = "reference",
            text_key = None,
            json_key = None,
            score_threshold = 5,
            minimum_image_size = 768,
        ):
        super().__init__()
        dataroot = osp.abspath(dataroot)
        self.sketch_root = osp.join(dataroot, control_key)
        self.color_root = osp.join(dataroot, image_key)
        self.ref_root = osp.join(dataroot, condition_key)
        self.text_root = osp.join(dataroot, text_key) if exists(text_key) else None

        self.load_image_list(dataroot, json_key, score_threshold, minimum_image_size)
        self.data_size = len(self)
        self.offset = random.randint(1, self.data_size) if mode == 'validation' else 0

    def load_image_list(self, dataroot, json_key, score_threshold, minimum_image_size):
        if exists(json_key):
            img_dict = {}
            json_files = glob(osp.join(dataroot, f"{json_key}*.json"))
            for jf in json_files:
                jf = osp.join(dataroot, jf)
                assert osp.exists(jf), f"Json file {jf} doesn't exist."
                d = json.load(open(jf, "r"))
                img_dict.update(d)

            self.data_items = [
                osp.join(self.sketch_root, file) for file in img_dict
                if check_json(img_dict[file], score_threshold, minimum_image_size)
            ]

        else:
            self.data_items = [
                file for ext in IMAGE_EXTENSIONS
                for file in
                glob(osp.join(self.sketch_root, f'*.{ext}')) + glob(osp.join(self.sketch_root, f'*/*.{ext}'))
            ]

    def get_images(self, index):
        filename = self.data_items[index]

        ske = Image.open(filename).convert('RGB')
        col = Image.open(filename.replace(self.sketch_root, self.color_root)).convert('RGB')
        w, h = col.size

        if self.offset > 0:
            ref_file = self.data_items[(index + self.offset) % self.data_size]
            ref = Image.open(ref_file.replace(self.sketch_root, self.color_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.sketch_root, self.ref_root)).convert('RGB')

        return {
            "control": ske,
            "image": col,
            "reference": [ref, None],
            "size": [h, w],
        }

    def __getitem__(self, index):
        return self.get_images(index)


    def __len__(self):
        return len(self.data_items)


class ZipTripletDataset(TripletDataset):
    def load_image_list(self, dataroot, json_key, score_threshold, minimum_image_size):
        if exists(json_key):
            img_dict = {}
            json_files = glob(osp.join(dataroot, f"{json_key}*.json"))
            for jf in json_files:
                jf = osp.join(dataroot, jf)
                assert osp.exists(jf), f"Json file {jf} doesn't exist."
                d = json.load(open(jf, "r"))
                img_dict.update(d)

            self.data_items = [
                (
                    file, 
                    img_dict[file].get("aesthetic_score", DEFAULT_AESTHETIC_SCORE),
                    img_dict[file].get("tags", "")
                )
                for file in img_dict if check_json(img_dict[file], score_threshold, minimum_image_size)
            ]

            # Add high-resolution generated image
            # self.data_items += [(file, DEFAULT_AESTHETIC_SCORE) for zip in glob(osp.join(self.sketch_root, "ai*.zip"))
            #                      for file in zipfile.ZipFile(zip).namelist() if not file.endswith("/")]

        else:
            self.data_items = [(file, DEFAULT_AESTHETIC_SCORE, "") for zip in glob(osp.join(self.sketch_root, "*.zip"))
                                for file in zipfile.ZipFile(zip).namelist() if not file.endswith("/")]

    def fresh_zip_files(self, zipid):
        self.zip_sketch, self.zip_color, self.zip_text = map(
            lambda t: zipfile.ZipFile(osp.join(t, f"{zipid}.zip"), "r") if exists(t) else None,
            (self.sketch_root, self.color_root, self.text_root)
        )

    def get_images(self, index):
        filename, aesthetic_score, tags = self.data_items[index]
        self.fresh_zip_files(osp.dirname(filename))

        ske = Image.open(self.zip_sketch.open(filename, "r")).convert('RGB')
        col = Image.open(self.zip_color.open(filename, "r")).convert('RGB')
        w, h = col.size

        if self.offset > 0:
            filename, aesthetic_score, rtags = self.data_items[(index + self.offset) % self.data_size]
            self.fresh_zip_files(osp.dirname(filename))
        ref = Image.open(self.zip_color.open(filename, "r")).convert('RGB')

        if exists(self.zip_text):
            text = self.zip_text.open(filename, "r").read().decode('utf-8')
        else:
            text = None

        return {
            "control": ske,
            "image": col,
            "reference": [ref, None],
            "size": [h, w],
            "text": text if exists(text) else tags,
            "aesthetic_score": torch.tensor(aesthetic_score)
        }

    def __del__(self):
        try:
            if exists(self.zip_color) and exists(self.zip_sketch):
                self.zip_sketch.close()
                self.zip_color.close()
        finally:
            self.zip_sketch = None
            self.zip_color = None


class QuartetDataset(ZipTripletDataset):
    def __init__(
            self,
            dataroot,
            *args,
            **kwargs
    ):
        super().__init__(dataroot=dataroot, *args, **kwargs)
        self.mask_root = osp.join(osp.abspath(dataroot), "mask")

    def fresh_zip_files(self, zipid):
        self.zip_sketch, self.zip_color, self.zip_mask, self.zip_text = map(
            lambda t: zipfile.ZipFile(osp.join(t, f"{zipid}.zip"), "r") if exists(t) else None,
            (self.sketch_root, self.color_root, self.mask_root, self.text_root)
        )

    def get_images(self, index):
        filename, aesthetic_score, tags = self.data_items[index]
        self.fresh_zip_files(osp.dirname(filename))

        ske = Image.open(self.zip_sketch.open(filename, "r")).convert('RGB')
        col = Image.open(self.zip_color.open(filename, "r")).convert('RGB')
        mask = Image.open(self.zip_mask.open(filename, "r")).convert('L')
        w, h = col.size

        if self.offset > 0:
            filename, aesthetic_score, rtags = self.data_items[(index + self.offset) % self.data_size]
            self.fresh_zip_files(osp.dirname(filename))
        ref = Image.open(self.zip_color.open(filename, "r")).convert('RGB')

        if exists(self.zip_text):
            text = self.zip_text.open(filename, "r").read().decode('utf-8')
        else:
            text = None

        return {
            "control": ske,
            "image": col,
            "reference": [ref, None],
            "smask": mask,
            "rmask": mask,
            "size": [h, w],
            "text": text if exists(text) else tags,
            "aesthetic_score": torch.tensor(aesthetic_score)
        }

    def __getitem__(self, index):
        while True:
            try:
                return self.get_images(index)

            except Exception as e:
                index += 1
                print(f"Cannot open file {self.data_items[index]} due to {e} !!!")
    
    def __del__(self):
        try:
            if exists(self.zip_color) and exists(self.zip_sketch) and exists(self.zip_mask):
                self.zip_sketch.close()
                self.zip_color.close()
                self.zip_mask.close()
        finally:
            self.zip_sketch = None
            self.zip_color = None
            self.zip_mask = None


class CharacterDataset(QuartetDataset):
    def __init__(self, dual_reference=True, *args, **kwargs):
        self.dual_reference = dual_reference
        super().__init__(*args, **kwargs)

    def fresh_zip_files(self, zipid):
        super().fresh_zip_files(zipid)
        while True:
            refname = self.zip_color.namelist()[random.randint(1, len(self.zip_color.namelist())-1)]
            if not refname.endswith("/"):
                return refname

    def get_images(self, index):
        filename, _, tags = self.data_items[index]
        refname = self.fresh_zip_files(osp.dirname(filename))
        refname = refname or filename

        ske = Image.open(self.zip_sketch.open(filename, "r")).convert('RGB')
        col = Image.open(self.zip_color.open(filename, "r")).convert('RGB')
        smask = Image.open(self.zip_mask.open(filename, "r")).convert('L')
        w, h = col.size

        ref = Image.open(self.zip_color.open(refname, "r")).convert('RGB')
        rmask = Image.open(self.zip_mask.open(refname, "r")).convert("L")
        rw, rh = ref.size

        return {
            "control": ske,
            "image": col,
            "reference": [ref, col] if self.dual_reference else [ref, None],
            "smask": smask,
            "rmask": rmask,
            "size": [h, w, rw, rh],
            "aesthetic_score": torch.tensor(DEFAULT_AESTHETIC_SCORE)
        }


class CustomCollateFn:
    """
        This class implements multi-resolution preprocessing.
    """
    def __init__(
            self,
            load_size = 768,
            crop_size = 768,
            ref_load_size = None,
            center_crop_max = 201,
            transform_list = {},
            keep_ratio = False,
            inverse_grayscale = False,
            eval_load_size = 768,
            erase_p = 0.,
            mask_expansion_p = 0.,
            mask_expansion_size = (60, 40),
            eval = False
    ):
        self.inverse_grayscale = inverse_grayscale

        if not eval:
            self.mask_expansion_p = mask_expansion_p
            self.erase_p = erase_p
            self.mask_expansion_size = mask_expansion_size
            self.preprocess = self.training_preprocess

            rotate = transform_list["rotate"]
            transform_list["rotate"] = False
            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)

            self.gt_seeds = partial(get_transform_seeds, transforms, load_size, crop_size)
            self.gt_transforms = partial(custom_transform, t=transforms)

            transform_list["rotate"] = rotate
            named_transforms = namedtuple('Transforms', transform_list.keys())
            transforms = named_transforms(**transform_list)
            ref_load_size = ref_load_size or load_size
            self.ref_seeds = partial(get_transform_seeds, transforms, ref_load_size, None)
            self.ref_transforms = partial(custom_transform, t=transforms)
            self.center_crop_max = center_crop_max
            self.image_size = crop_size

        else:
            self.preprocess = self.testing_preprocess
            self.image_size = eval_load_size if eval_load_size else crop_size
            self.kr = keep_ratio

    def training_preprocess(self, batch):
        gt_seeds = self.gt_seeds()
        ref_seeds = self.ref_seeds()
        load_size = gt_seeds[-1]

        inputs = {"crop": []}
        for k in batch[0].keys():
            inputs[k] = [item[k] for item in batch]
        for i in range(len(batch)):
            item = batch[i]
            ske = item["control"]
            col = item["image"]
            ref, oref = item["reference"]

            ske = self.gt_transforms(ske, *gt_seeds, center_crop_max=self.center_crop_max)
            col, ct, cl, rh, rw = self.gt_transforms(col, *gt_seeds, crop_ratio=True)
            ref = self.ref_transforms(ref, *ref_seeds)

            inputs["control"][i] = -normalize(ske) if self.inverse_grayscale else normalize(ske)
            inputs["image"][i] = normalize(col)
            inputs["size"][i] = torch.Tensor(item["size"] + [load_size])
            inputs["crop"].append(torch.Tensor([ct, cl, rh, rw]))

            smask = item.get("smask", None)
            rmask = item.get("rmask", None)

            if exists(smask) and exists(rmask):
                smask = to_tensor(self.gt_transforms(smask, *gt_seeds))
                rmask = to_tensor(self.ref_transforms(rmask, *ref_seeds))

                if self.mask_expansion_p > 0 and random.rand() < self.mask_expansion_p:
                    rmask = mask_expansion(rmask, *self.mask_expansion_size)
                elif self.erase_p > 0 and random.rand() < self.erase_p:
                    rmask = transforms.RandomErasing(
                        p=0.5, scale=(0.2, 0.5), ratio=(0.2, 3), value=1, inplace=True
                    )(rmask)
                else:
                    rmask = F.max_pool2d(rmask, kernel_size=21, stride=1, padding=10)

                inputs["smask"][i] = smask
                inputs["rmask"][i] = rmask

            if exists(oref):
                # dual reference for character-specified training
                oref = self.ref_transforms(oref, *ref_seeds)
                inputs["reference"][i] = torch.cat([normalize(ref), normalize(oref)])
                omask = item.get("smask", None)

                if exists(omask):
                    omask = to_tensor(self.ref_transforms(omask, *ref_seeds))
                    if self.mask_expansion_p > 0 and random.rand() < self.mask_expansion_p:
                        omask = mask_expansion(omask, *self.mask_expansion_size)
                    elif self.erase_p > 0 and random.rand() < self.erase_p:
                        omask = transforms.RandomErasing(
                            p=0.5, scale=(0.1, 0.2), ratio=(0.33, 3), value=1, inplace=True
                        )(omask)
                    else:
                        rmask = F.max_pool2d(rmask, kernel_size=21, stride=1, padding=10)

                    inputs["rmask"][i] = torch.cat([rmask, omask])

            else:
                inputs["reference"][i] = normalize(ref)

        for k in inputs:
            if k not in ["tag", "text"]:
                inputs[k] = torch.stack(inputs[k])
        return inputs

    def testing_preprocess(self, batch):
        resize = resize_with_ratio if self.kr else resize_without_ratio
        inputs = {}

        for k in batch[0].keys():
            inputs[k] = [item[k] for item in batch]

        for i in range(len(batch)):
            item = batch[i]

            ske = resize_and_pad(item["control"], self.image_size, self.image_size)
            col = resize(item["image"], self.image_size)
            ref = resize(item["reference"][0], self.image_size)
            inputs["control"][i] = -normalize(ske) if self.inverse_grayscale else normalize(ske)
            inputs["image"][i] = normalize(col)
            inputs["reference"][i] = normalize(ref)
            inputs["size"][i] = torch.Tensor(item["size"] + [self.image_size])

        for k in inputs:
            inputs[k] = torch.stack(inputs[k])
        return inputs

    def __call__(self, batch):
        return self.preprocess(batch)


def create_dataloader(opt, cfg, device_num, eval_load_size=None):
    DATALOADER = {
        "TripletLoader": TripletDataset,
        "ZipTripletLoader": ZipTripletDataset,
        "QuartLoader": QuartetDataset,
        "CharacterLoader": CharacterDataset,
    }

    loader_cls = cfg["class"]
    assert loader_cls in DATALOADER.keys(), f"DataLoader {loader_cls} does not exist."
    loader = DATALOADER[loader_cls]

    dataset = loader(
        mode            = opt.mode,
        dataroot        = opt.dataroot,
        **cfg["dataset_params"]
    )
    custom_collate = CustomCollateFn(eval=opt.eval, eval_load_size=eval_load_size, **cfg["transforms"])

    dataloader = data.DataLoader(
        dataset         = dataset,
        batch_size      = opt.batch_size,
        shuffle         = cfg.get("shuffle", True) and not opt.eval,
        num_workers     = opt.num_threads,
        drop_last       = device_num > 1,
        pin_memory      = True,
        prefetch_factor = 2 if opt.num_threads > 0 else None,
        collate_fn      = custom_collate,
    )
    return dataloader, len(dataset)
