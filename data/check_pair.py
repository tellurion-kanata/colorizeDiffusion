import os
import argparse
import multiprocessing
import PIL.Image as Image

from glob import glob


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--num_threads', '-nt', type=int, default=32)
    parser.add_argument('--target', '-t', type=str, default='original')
    parser.add_argument('--not_delete', '-nd', action='store_true')
    return parser.parse_args()

def processing(thread_id, img_files, target_root, pair_root, delete=True):
    data_size = len(img_files)

    for i in range(data_size):
        pair_file = img_files[i].replace(target_root, pair_root)
        delete_flag = ", deleted." if delete else "."

        if not (os.path.exists(img_files[i]) and os.path.exists(pair_file)):
            print(f"{img_files[i]} does not have a pair image in {pair_root}{delete_flag}")
            if delete:
                os.remove(img_files[i])
        else:
            try:
                img = Image.open(img_files[i]).convert("RGB")
                img.save(img_files[i])
            except:
                print(f"cannot convert {img_files[i]} to RGB format{delete_flag}")

        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    target_keys = ["sketch", "original"]
    dataroot = opt.dataroot
    target_keys.remove(opt.target)

    target_root = os.path.join(dataroot, opt.target)
    pair_root = os.path.join(dataroot, target_keys[0])
    img_files = [file for file in glob(os.path.join(target_root, '*')) if not os.path.isdir(file)]
    img_files += [file for file in glob(os.path.join(target_root, '*/*')) if not os.path.isdir(file)]
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
                thread = multiprocessing.Process(
                    target=processing,
                    args=(t, img_files[t*thread_size: ], target_root, pair_root, not opt.not_delete)
                )
            else:
                thread = multiprocessing.Process(
                    target=processing,
                    args=(t, img_files[t*thread_size: (t+1)*thread_size], target_root, pair_root, not opt.not_delete)
                )
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
