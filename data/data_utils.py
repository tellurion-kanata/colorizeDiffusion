import os
import cv2
import argparse
import threading
import numpy as np
import numpy.random as random
import PIL.Image as Image

from glob import glob

# resize cmd:  nohup python -u data_utils.py -d /home2/danbooru/ -s /home2/danbooru/ --resize --nsize 512 &

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

DELETE = True
MAX_IMAGE_SIZE = 89478485

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--save_path', '-s', type=str, default=None, help='save path')
    parser.add_argument('--resize', action='store_true', help='resize image')
    parser.add_argument('--padding', action='store_true', help='pad image before resize')
    parser.add_argument('--warp', action='store_true', help='deform image')
    parser.add_argument('--spray', action='store_true', help='draw thick line on the image')
    parser.add_argument('--blur', action='store_true', help='use mean filter to blur the image')
    parser.add_argument('--mosaic', action='store_true', help='add mosaic to the image')
    parser.add_argument('--noise', action='store_true', help='add gaussian noise')
    parser.add_argument('--nsize', type=int, help='if nize is not given, resize only pad image with white margin')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha used for deformation')
    parser.add_argument('--density', default=1.0, type=float, help='density used for deformation')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--to_rgb', action='store_true', help='normalize images to rgb format')
    parser.add_argument('--num_threads', type=int, default=8)
    return parser.parse_args()


""" Implemented by Jarvis73 - Jiawei Zhang """
""" Github link: https://github.com/Jarvis73/Moving-Least-Squares """
def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    t = np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha
    # w = 1.0 / t # [ctrls, grow, gcol]
    w = np.divide(1., t, out=np.ones_like(t), where=(t!=0))
    sum_w = np.sum(w, axis=0)  # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)  # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                  reshaped_mul_right.transpose(0, 3, 4, 1, 2))  # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / sum_w  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)  # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)  # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)  # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)  # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar + qstar  # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image


def get_random_points(h, w, num=4, scope=75):
    p = random.randint(0, min(h, w), [num, 2])
    q = np.zeros([num, 2])
    for i in range(num):
        dd = [0, 0]
        dd[0] = random.randint(max(-scope, -p[i, 0]), min(scope, h - p[i, 0] - 1))
        dd[1] = random.randint(max(-scope, -p[i, 1]), min(scope, w - p[i, 1] - 1))
        q[i] = p[i] + dd
    return p, q


def warp(img, alpha, density, warp_count=2):
    h, w = img.shape[:2]
    for i in range(warp_count):
        p, q = get_random_points(h, w)
        img = mls_rigid_deformation(img, p, q, alpha, density)

    return img


def resize_and_padding(img, padding=False, nsize=None):
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
    return padded


def drawLine(img, width_scale=0.2, color_ts=740):
    h, w, c = img.shape
    base = max(h, w)
    width = int(base * width_scale)
    width = width + width % 2
    mx, my, len = 0, 0, 0
    while True:
        x1, x2 = random.randint(0, h, 2)
        y1, y2 = random.randint(0, w, 2)
        len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if x1 == x2 and y1 == y2 and len < base * 0.75:
            continue
        mx, my = (x1+x2) // 2, (y1+y2) // 2
        if img[mx, my].sum() < color_ts:
            break
    bgr = (img[mx, my][0].item(), img[mx, my][1].item(), img[mx, my][2].item())
    dir_x = (x2 - x1) / len
    dir_y = (y2 - y1) / len

    radius = width // 2
    stroke_color = np.full((h, w, c), bgr, dtype=np.uint8)
    beta = np.zeros((h, w))
    for i in range(int(len)):
         center = (round(x1+dir_x), round(y1+dir_y))
         x1 += dir_x
         y1 += dir_y
         cv2.circle(beta, center, radius=radius, color=1, thickness=radius)
    center = width // 2
    kernel = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            dx, dy = x - center, y - center
            kernel[y, x] = np.exp(-(dx*dx+dy*dy) / (500*center))
    kernel /= kernel.sum()
    beta = cv2.filter2D(beta, -1, kernel) * 0.975
    beta = np.expand_dims(beta, 2).repeat(3, axis=2)
    alpha = 1. - beta

    dst = (alpha * img + beta * stroke_color).astype(np.uint8)
    return dst


def spray(img, stroke_num=2):
    for i in range(stroke_num):
        img = drawLine(img)
    return img


def mosaic(img):
    """
    :param rgb_img
    left upper point: (sx, sy)
    mosaic region height and width: (kh, kw)
    size of each mosaic: neighbor
    """
    h, w, c = img.shape

    result = img.copy()
    num = np.random.randint(8, 16)
    for i in range(num):
        neighbor = np.random.randint(6, 12)
        kh, kw = np.random.randint(40, 70, 2)
        sx, sy = np.random.randint(0, h-kh), np.random.randint(0, w-kw)
        for dx in range(0, kh, neighbor):
            for dy in range(0, kw, neighbor):
                rect = [sx + dx, sy + dy]
                color = img[sy+dy][sx+dx].tolist()

                tx = rect[0] + neighbor - 1
                ty = rect[1] + neighbor - 1
                tx = min(tx, tx + kh)
                ty = min(ty, sy + kw)

                left_up = (rect[0], rect[1])
                right_down = (tx,ty)
                cv2.rectangle(result, left_up, right_down, color, -1)
    return result


def gaussian_noise(img, mean=0):
    img = img / 255.
    var = random.random(1) / 50.
    noise = random.normal(mean, var**0.5, img.shape)
    img += noise
    img = img.clip(0, 1.) * 255.
    return img.astype(np.uint8)


def blur(img, min_ks=1, max_ks=10):
    ks = random.randint(min_ks, max_ks)
    return cv2.blur(img, (ks, ks))


def check_delete(img):
    try:
        h, w, c = img.shape
        if h / w > 4 or w / h > 4:
            return True
        return False
    except:
        return True

def processing(thread_id, opt, img_files):
    dataroot = os.path.abspath(opt.dataroot)
    data_size = len(img_files)
    save_path = os.path.abspath(opt.save_path) if opt.save_path is not None else None
    alpha = opt.alpha
    density = opt.density

    if not opt.spray and not opt.warp and not opt.resize and not opt.mosaic and not opt.noise and not opt.check and \
            not opt.to_rgb:
        raise ModuleNotFoundError('Please select a pre-processing util.')
    for i in range(data_size):
        if opt.to_rgb:
            try:
                img = Image.open(img_files[i]).convert('RGB')
                if img.size[0] * img.size[1] > MAX_IMAGE_SIZE:
                    print(f"cannot convert {img_files[i]} to rgb, deleted")
                    os.remove(img_files[i])
                    continue
                img.save(img_files[i])
            except:
                print(f"cannot convert {img_files[i]} to rgb, deleted")
                os.remove(img_files[i])
        else:
            img = cv2.imread(img_files[i])
            try:
                if opt.spray:
                    img = spray(img)
                if opt.blur:
                    img = blur(img)
                if opt.mosaic:
                    img = mosaic(img)
                if opt.noise:
                    img = gaussian_noise(img)
                if opt.resize or opt.padding:
                    img = resize_and_padding(img, opt.padding, opt.nsize)
                if opt.warp:
                    img = warp(img, alpha, density)
            except:
                if DELETE:
                    print(img_files[i])
                    os.remove(img_files[i])

            if opt.check:
                if check_delete(img):
                    print(img_files[i])
                    os.remove(img_files[i])
            else:
                filename = os.path.abspath(img_files[i])
                sname = filename if save_path is None else filename.replace(dataroot, save_path)
                basename = os.path.basename(filename)
                os.makedirs(sname.replace(basename, ''), exist_ok=True)
                cv2.imwrite(sname, img)
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    dataroot = opt.dataroot
    save_path = opt.save_path
    if save_path is not None and not os.path.exists(save_path):
        os.mkdir(save_path)

    img_files = [file for ext in IMAGE_EXTENSIONS
                   for file in glob(os.path.join(dataroot, '*.{}'.format(ext)))]
    img_files += [file for ext in IMAGE_EXTENSIONS
                  for file in glob(os.path.join(dataroot, '*/*.{}'.format(ext)))]
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
