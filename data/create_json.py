import os.path as osp
import argparse
import multiprocessing
import zipfile
import json
import PIL.Image as Image

from glob import glob


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--num_threads', '-nt', type=int, default=32)
    parser.add_argument('--target', '-t', type=str, default='original')
    parser.add_argument('--data_json', type=str, default='dataset')
    return parser.parse_args()

def get_zip(zip_path, filename):
    return zipfile.ZipFile(osp.join(zip_path, f"{osp.dirname(filename)}.zip"), "r")

def processing(thread_id, zip_path, img_files, pair_files, json_name):
    data_size = len(img_files)

    files = []
    for idx, filename in enumerate(img_files):
        try:
            if filename in pair_files:
                image = Image.open(get_zip(zip_path, filename).open(filename))
                if image.size[0] * image.size[1] < Image.MAX_IMAGE_PIXELS:
                    files.append(filename)
                else:
                    print(f"{filename} exceeds image size limitation")
        except:
            print(f"{filename} is not included in original images or not readable.")
        finally:
            if idx % 5000 == 0:
                print(f"process {thread_id}: [{idx + 1}/{data_size}]")


    with open(f"{json_name}_{thread_id}.json", "w") as f:
        json.dump(files, f, indent=0)


def create_threads(opt):
    target_keys = ["sketch", "original"]
    dataroot = opt.dataroot
    json_name = opt.data_json
    target_keys.remove(opt.target)

    target_root = osp.abspath(osp.join(dataroot, opt.target))
    pair_root = osp.abspath(osp.join(dataroot, target_keys[0]))

    sketch_zips = glob(osp.join(target_root, "*.zip"))
    img_files = []
    for zip in sketch_zips:
        img_files += zipfile.ZipFile(zip, "r").namelist()

    pair_zips = glob(osp.join(pair_root, "*.zip"))

    pair_files = []
    for zip in pair_zips:
        pair_files += zipfile.ZipFile(zip, "r").namelist()

    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, pair_root, img_files, target_root, pair_root)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = multiprocessing.Process(
                    target=processing,
                    args=(t, pair_root, img_files[t*thread_size: ], pair_files, json_name)
                )
            else:
                thread = multiprocessing.Process(
                    target=processing,
                    args=(t, pair_root, img_files[t*thread_size: (t+1)*thread_size], pair_files, json_name)
                )
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print("process finished.")


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)