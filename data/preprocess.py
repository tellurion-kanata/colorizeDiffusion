import os
import cv2
import argparse
import threading
import numpy as np
import os.path as osp
import PIL.Image as Image
import PIL.ImageFile as ImageFile

from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

DELETE = True
MAX_IMAGE_SIZE = 89478485


"""
    python -u preprocess.py -d path --resize --nsize 512 --check 
"""

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--save_path', '-s', type=str, default=None, help='save path')
    parser.add_argument('--resize', action='store_true', help='resize image')
    parser.add_argument('--padding', action='store_true', help='pad image before resize')
    parser.add_argument('--nsize', type=int, help='if nize is not given, resize only pad image with white margin')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--num_threads', '-nt', type=int, default=8)
    return parser.parse_args()


def resize_and_padding(img, padding=False, nsize=None):
    img = np.array(img)
    h, w, c = img.shape
    base = max(h, w)

    padded = img
    if padding:
        padded = np.full((base, base, 3), 255, dtype=np.uint8)
        padded[(base - h) // 2:(base + h) // 2, (base - w) // 2:(base + w) // 2, :] = img

    if nsize is None:
        return padded
    if h > w:
        padded = cv2.resize(padded, (nsize, int(nsize * h / w)))
    else:
        padded = cv2.resize(padded, (int(nsize * w / h), nsize))
    return Image.fromarray(padded)


def check_delete(img):
    try:
        h, w, c = img.shape
        if h / w > 4 or w / h > 4:
            return True
        return False
    except:
        return True

def processing(thread_id, opt, dataroot, save_path, img_files):
    data_size = len(img_files)

    for i in range(data_size):
        try:
            img = Image.open(img_files[i]).convert('RGB')
            h, w = img.size[:]
            if opt.check and (h * w > MAX_IMAGE_SIZE or h / w > 4 or w / h > 4):
                print(f"cannot convert {img_files[i]} to rgb or the aspect ratio exceeds, deleted")
                os.remove(img_files[i])
            else:
                filename = img_files[i]
                if opt.resize:
                    img = resize_and_padding(img, opt.padding, opt.nsize)
                if save_path is not None:
                    filename = filename if save_path is None else filename.replace(dataroot, save_path)
                    os.makedirs(filename.replace(osp.basename(filename), ''), exist_ok=True)
                img.save(filename)
        except:
            print(f"cannot convert {img_files[i]}, deleted")
            os.remove(img_files[i])
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    dataroot = osp.abspath(opt.dataroot)
    save_path = osp.abspath(opt.save_path) if opt.save_path is not None else None

    img_files = [file for ext in IMAGE_EXTENSIONS
                   for file in glob(osp.join(dataroot, '*.{}'.format(ext)))]
    img_files += [file for ext in IMAGE_EXTENSIONS
                  for file in glob(osp.join(dataroot, '*/*.{}'.format(ext)))]
    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, opt, dataroot, save_path, img_files)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = threading.Thread(
                    target=processing,
                    args=(t, opt, dataroot, save_path, img_files[t*thread_size: ])
                )
            else:
                thread = threading.Thread(
                    target=processing,
                    args=(t, opt, dataroot, save_path, img_files[t*thread_size: (t+1)*thread_size])
                )
            threads.append(thread)
        for t in threads:
            t.start()
        thread.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
