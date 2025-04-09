"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""



import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    bic = Image.Resampling.BICUBIC
except ImportError:
    bic = Image.BICUBIC

import os
import numpy as np
import argparse
import functools

from tqdm import tqdm
from os.path import *
from glob import glob
from preprocessor.anime2sketch import UnetGenerator


bic = Image.Resampling.BICUBIC

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    """if a given filename is a valid image
    Parameters:
        filename (str) -- image filename
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(path):
    """read the paths of valid images from the given directory path
    Parameters:
        path (str)    -- input directory path
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_transform(load_size=0, grayscale=False, method=bic, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if load_size > 0:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def read_img_path(path, load_size):
    """read tensors from a given image path
    Parameters:
        path (str)     -- input image path
        load_size(int) -- the input size. If <= 0, don't resize
    """
    img = Image.open(path).convert('RGB')
    aus_resize = None
    if load_size > 0:
        aus_resize = img.size
    transform = get_transform(load_size=load_size)
    image = transform(img)
    return image.unsqueeze(0), aus_resize

def tensor_to_img(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, output_resize=None):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array)    -- input numpy array
        image_path (str)             -- the path of the image
        output_resize(None or tuple) -- the output size. If None, don't resize
    """

    image_pil = Image.fromarray(image_numpy)
    if output_resize is not None:
        image_pil = image_pil.resize(output_resize, bic)
    image_pil.save(image_path)

class ImageDataset(data.Dataset):
    def __init__(self,
                 dataroot,
                 load_size=512,):
        super().__init__()
        self.dataroot = dataroot
        self.image_files = [file for ext in IMAGE_EXTENSIONS
                            for file in glob(join(dataroot, '*.{}'.format(ext)))]
        self.image_files += [file for ext in IMAGE_EXTENSIONS
                             for file in glob(join(dataroot, '*/*.{}'.format(ext)))]
        self.load_size = load_size
        self.transform = transforms.Compose([
            transforms.Resize([load_size, load_size], bic),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        while True:
            try:
                image_file = self.image_files[index]
                img = Image.open(image_file).convert('RGB')
                path = image_file
                h, w = img.size

                img = self.transform(img)
                return {'color': img,
                        'path': path,
                        'height': h,
                        'width': w}
            except Exception as e:
                print(f"Cannot handle file {self.image_files[index]} for sketch extracting due to {e}, deleting..")
                os.remove(self.image_files[index])
                index += 1

    def __len__(self):
        return len(self.image_files)


def create_model():
    """Create a model for anime2sketch
    hardcoding the options for simplicity
    """
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    ckpt = torch.load('weights/netG.pth', weights_only=True)
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    net.load_state_dict(ckpt)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument('--dataroot','-i', default='test_samples/', type=str)
    parser.add_argument('--load_size','-s', default=512, type=int)
    parser.add_argument('--batch_size', '-bs', default=32, type=int)
    parser.add_argument('--num_threads', '-nt', default=8, type=int)
    parser.add_argument('--output_dir','-o', default='results/', type=str)
    parser.add_argument('--gpu_ids', '-g', default=[], help="gpu ids: e.g. 0 0,1,2 0,2.")
    opt = parser.parse_args()

    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if len(opt.gpu_ids)>0 else 'cpu')
    model = create_model().to(device)      # create a model given opt.model and other options
    model.eval()

    # save outputs
    dataroot = abspath(opt.dataroot)
    save_dir = abspath(opt.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataloader = data.DataLoader(
        dataset     = ImageDataset(dataroot, opt.load_size),
        batch_size  = opt.batch_size,
        num_workers = opt.num_threads
    )

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            img, paths, aus_height, aus_width = inputs['color'], inputs['path'], inputs['height'], inputs['width']
            # img,  aus_resize = read_img_path(test_path, opt.load_size)
            aus_tensors = model(img.to(device))
            for idx, tensor in enumerate(aus_tensors):
                aus_img = tensor_to_img(tensor)
                aus_path = paths[idx].replace(dataroot, save_dir)
                filename = basename(paths[idx])
                os.makedirs(aus_path.replace(filename, ''), exist_ok=True)
                save_image(aus_img, aus_path, (aus_height[idx], aus_width[idx]))