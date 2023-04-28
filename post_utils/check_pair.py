import os
import PIL.Image as Image
from glob import glob

color_root = "/home2/danbooru/color"
tag_root = "/home2/danbooru/tags"

tag_files = glob(os.path.join(tag_root, "*/*.json"))
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}


for file in tag_files:
    flag = False
    for ext in IMAGE_EXTENSIONS:
        try:
            Image.open(file.replace(tag_root, color_root).replace('json', ext))
            flag = True
            break
        except:
            continue
    if not flag:
        print(f"cannot find corresponding color image for {file}")
