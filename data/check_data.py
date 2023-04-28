import os
import argparse
import threading
import PIL.Image as Image

from glob import glob


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--num_threads', type=int, default=8)
    return parser.parse_args()

def processing(thread_id, opt, img_files):
    dataroot = os.path.abspath(opt.dataroot)
    data_size = len(img_files)

    sketch_root = os.path.join(dataroot, 'sketch')
    color_root = os.path.join(dataroot, 'color')
    
    for i in range(data_size):
        if not (os.path.exists(img_files[i]) and os.path.exists(img_files[i].replace(sketch_root, color_root))):
            print(f"{img_files[i]} does not have a pair image in color files, deleted")
            os.remove(img_files[i])
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    dataroot = opt.dataroot

    sketch_root = os.path.join(dataroot, 'sketch')
    img_files = [file for ext in IMAGE_EXTENSIONS
                   for file in glob(os.path.join(sketch_root, '*.{}'.format(ext)))]
    img_files += [file for ext in IMAGE_EXTENSIONS
                  for file in glob(os.path.join(sketch_root, '*/*.{}'.format(ext)))]
    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, opt, img_files)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size: ]))
            else:
                thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size: (t+1)*thread_size]))
            threads.append(thread)
        for t in threads:
            t.start()
        thread.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
