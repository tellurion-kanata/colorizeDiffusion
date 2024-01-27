import os
import zipfile
import argparse
import multiprocessing
import numpy as np
import os.path as osp
import PIL.Image as Image

from glob import glob

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

DELETE = True


"""
    python -u preprocess.py -d path --resize --nsize 512 --check 
"""

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--save_path', '-s', type=str, default=None, help='save path')
    parser.add_argument('--resize', action='store_true', help='resize image')
    parser.add_argument('--padding', action='store_true', help='pad image before resize')
    parser.add_argument('--nsize', type=int, default=2048, help='if nize is not given, resize only pad image with white margin')
    parser.add_argument('--not_delete', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--num_workers', '-nw', type=int, default=8)
    return parser.parse_args()


def padding(img):
    img = np.array(img)
    h, w, c = img.shape
    base = max(h, w)

    padded = np.full((base, base, 3), 255, dtype=np.uint8)
    padded[(base - h) // 2:(base + h) // 2, (base - w) // 2:(base + w) // 2, :] = img
    return Image.fromarray(padded)


def resize_with_ratio(img: Image.Image, w, h, nsize=None):
    if w > h:
        img = img.resize((int(w / h * nsize), nsize), Image.BICUBIC)
    else:
        img = img.resize((nsize, int(h / w * nsize)), Image.BICUBIC)
    return img


def check_delete(img):
    try:
        h, w, c = img.shape
        if h / w > 4 or w / h > 4:
            return True
        return False
    except:
        return True


def processing(thread_id, opt, dataroot, save_path, img_files, delete=True):
    data_size = len(img_files)

    for i in range(data_size):
        try:
            filename = img_files[i]
            # img = Image.open(img_files[i]).convert('RGB')
            img = Image.open(zipfile.ZipFile(
                osp.join(dataroot, f"{osp.dirname(filename)}.zip"), "r"
            ).open(filename, "r")).convert("RGB")

            w, h = img.size[:]
            if opt.check and (h * w > Image.MAX_IMAGE_PIXELS or h / w > 4 or w / h > 4):
                print(f"{img_files[i]} exceeds the aspect ratio, delete: {delete}")
                # if delete:
                #     os.remove(img_files[i])
            else:
                if opt.padding:
                    img = padding(img)
                if opt.resize and h > opt.nsize and w > opt.nsize:
                    img = resize_with_ratio(img, w, h, opt.nsize)

                save_name = osp.join(save_path, filename)
                dirname = osp.dirname(save_name)
                if not osp.exists(dirname):
                    os.makedirs(dirname, exist_ok=True)
                img.save(save_name)
        except:
            print(f"cannot convert {img_files[i]} to rgb, delete: {delete}")
            # if delete:
            #     os.remove(img_files[i])
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_workers(opt):
    dataroot = osp.abspath(opt.dataroot)
    save_path = osp.abspath(opt.save_path) if opt.save_path is not None else dataroot

    # saved in directories
    # img_files = [file for ext in IMAGE_EXTENSIONS
    #                for file in glob(osp.join(dataroot, '*.{}'.format(ext)))]
    # img_files += [file for ext in IMAGE_EXTENSIONS
    #               for file in glob(osp.join(dataroot, '*/*.{}'.format(ext)))]

    zips = glob(osp.join(dataroot, "*.zip"))
    img_files = []
    for zip in zips:
        img_files += zipfile.ZipFile(zip, "r").namelist()

    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_workers = opt.num_workers

    if num_workers == 0:
        processing(0, opt, dataroot, save_path, img_files)
    else:
        thread_size = data_size // num_workers
        ps = []
        for t in range(num_workers):
            if t == num_workers - 1:
                p = multiprocessing.Process(
                    target=processing,
                    args=(t, opt, dataroot, save_path, img_files[t*thread_size: ], not opt.not_delete)
                )
            else:
                p = multiprocessing.Process(
                    target=processing,
                    args=(t, opt, dataroot, save_path, img_files[t*thread_size: (t+1)*thread_size], not opt.not_delete)
                )
            ps.append(p)
        for p in ps:
            p.start()
        for p in ps:
            p.join()
    print("process finished.")


if __name__ == '__main__':
    opt = get_options()
    create_workers(opt)
