import os
import argparse
import numpy as np
import torch
import cv2
import torch.utils.data as data
import PIL.Image as Image

from preprocessor.sketchKeras import SketchKeras
from glob import glob
from os.path import *
from tqdm import tqdm

"""
    Modified sketchKeras.
    Github: https://github.com/higumax/sketchKeras-pytorch
    Author: higumax, lllyasviel
"""


device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

bic = Image.Resampling.BICUBIC

def tensor_to_numpy(tensors):
    return tensors.permute(0, 1, 3, 4, 2).cpu().numpy()

def check_delete(h, w):
    try:
        if h / w > 4 or w / h > 4:
            return True
        return False
    except:
        return True


def save_image(output, image_path, height, width):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array)    -- input numpy array
        image_path (str)             -- the path of the image
        output_resize(None or tuple) -- the output size. If None, don't resize
    """
    output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, output)


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

    def __getitem__(self, index):
        while True:
            try:
                path = self.image_files[index]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                height, width = img.shape[:2]

                if check_delete(height, width):
                    raise Exception()

                if width > height:
                    new_width, new_height = (512, int(512 / width * height))
                else:
                    new_width, new_height = (int(512 / height * width), 512)
                img = cv2.resize(img, (new_width, new_height))

                h, w, c = img.shape
                blurred = cv2.GaussianBlur(img, (0, 0), 3)
                highpass = img.astype(int) - blurred.astype(int)
                highpass = highpass.astype(np.float32) / 127.5
                highpass /= np.max(highpass)

                img = np.zeros((512, 512, 3), dtype=np.float32)
                img[0:h, 0:w, 0:c] = highpass
                img = torch.tensor(img).permute(2, 0, 1)
                return {'color': img,
                        'path': path,
                        'height': height,
                        'width': width,
                        'new_height': new_height,
                        'new_width': new_width}
            except Exception as e:
                print(f"Cannot handle {self.image_files[index]} due to {e}, deleting...")
                os.remove(self.image_files[index])
                index += 1

    def __len__(self):
        return len(self.image_files)


def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument('--dataroot','-i', default='test_samples/', type=str)
    parser.add_argument('--load_size','-s', default=512, type=int)
    parser.add_argument('--batch_size', '-bs', default=32, type=int)
    parser.add_argument('--num_threads', '-nt', default=8, type=int)
    parser.add_argument('--output_dir','-o', default='results/', type=str)
    parser.add_argument('--gpu_ids', '-g', default=[], help="gpu ids: e.g. 0 0,1,2 0,2.")
    parser.add_argument('--model', default="weights/model.pth", help="gpu ids: e.g. 0 0,1,2 0,2.")
    opt = parser.parse_args()

    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if len(opt.gpu_ids)>0 else 'cpu')
    model = SketchKeras().to(device)      # create a model given opt.model and other options
    model.load_state_dict(torch.load(opt.model, weights_only=True))
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

    with (torch.no_grad()):
        for inputs in tqdm(dataloader):
            img, paths = inputs['color'], inputs['path']
            heights, widths, nhs, nws = inputs['height'], inputs['width'], inputs['new_height'], inputs['new_width']
            b, c, h, w = img.shape
            img = img.reshape(b*c, 1, h, w).contiguous()
            outputs = model(img.to(device))
            outputs = outputs.reshape(b, c, 1, h, w).contiguous()
            outputs = tensor_to_numpy(outputs)
            for idx, output in enumerate(outputs):
                new_height, new_width = nhs[idx], nws[idx]
                output = postprocess(output.squeeze(), thresh=0.1)
                output = output[:new_height, :new_width]
                aus_path = paths[idx].replace(dataroot, save_dir)
                filename = basename(paths[idx])
                os.makedirs(aus_path.replace(filename, ''), exist_ok=True)
                save_image(output, aus_path, int(heights[idx]), int(widths[idx]))