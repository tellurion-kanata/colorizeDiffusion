import os
import PIL.Image as Image
import json

from glob import glob
from tqdm import tqdm

color_root = "/raid/danbooru/sketch"
tag_root = "/raid/danbooru/tags"

tag_files = glob(os.path.join(tag_root, "*/*.json"))
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}


for file in tqdm(tag_files):
    flag = False
    for ext in IMAGE_EXTENSIONS:
        try:
            img_filename = file.replace(tag_root, color_root).replace('json', ext)
            Image.open(img_filename)
            flag = True

            if flag:
                with open(file, 'r') as f:
                    dict = json.load(f)
                with open(file, 'w') as f:
                    dict.update({"linked_img": img_filename})
                    json.dump(dict, f, indent=1)
            break
        except:
            continue
    if not flag:
        print(f"cannot find corresponding color image for {file}, deleted")
        # os.remove(file)