import os
import argparse
import threading
import PIL.Image as Image

from glob import glob


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--target', '-t', type=str, default='color')
    return parser.parse_args()

def processing(thread_id, img_files, target_root, pair_root):
    data_size = len(img_files)

    for i in range(data_size):
        if not (os.path.exists(img_files[i]) and os.path.exists(img_files[i].replace(target_root, pair_root))):
            print(f"{img_files[i]} does not have a pair image in {pair_root}, deleted")
            os.remove(img_files[i])
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    target_keys = ["sketch", "color"]
    dataroot = opt.dataroot
    target_keys.remove(opt.target)

    target_root = os.path.join(dataroot, opt.target)
    pair_root = os.path.join(dataroot, target_keys[0])
    img_files = [file for file in glob(os.path.join(target_root, '*'))]
    img_files += [file for file in glob(os.path.join(target_root, '*/*'))]
    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, img_files, target_root, pair_root)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = threading.Thread(
                    target=processing,
                    args=(t, img_files[t*thread_size: ], target_root, pair_root)
                )
            else:
                thread = threading.Thread(
                    target=processing,
                    args=(t, img_files[t*thread_size: (t+1)*thread_size], target_root, pair_root)
                )
            threads.append(thread)
        for t in threads:
            t.start()
        thread.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
