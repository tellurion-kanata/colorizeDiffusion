import os
import cv2
import random
import argparse
from glob import glob

def get_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--save_path', default='./sketch')
    return parser.parse_args()


def mix(opt):
    dataroot = os.path.join(opt.dataroot, 'pencil1')
    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files = glob(os.path.join(dataroot, '*.jpg'))
    data_size = len(files)
    i = 0
    for file in files:
        index = random.randint(1, 4)
        current_filename = file.replace('pencil1', 'pencil{}'.format(index))
        img = cv2.imread(current_filename)
        cv2.imwrite(file.replace(dataroot, save_path), img)
        i += 1
        print('process: [{} / {}]'.format(i, data_size))


if __name__ == '__main__':
    opt = get_option()
    mix(opt)