import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

from ldm.modules.encoders.modules import OpenCLIP, OPENAI_MEAN, OPENAI_STD

def gap(x: torch.Tensor = None, keepdim=True):
    if len(x.shape) == 4:
        return torch.mean(x, dim=[2, 3], keepdim=keepdim)
    elif len(x.shape) == 3 or len(x.shape) == 2:
        return torch.mean(x, dim=[1], keepdim=keepdim)
    else:
        raise NotImplementedError('gap input should be 3d or 4d tensors')

def maxmin(x: torch.Tensor, threshold=0.2):
    maxm = x.max()
    minm = x.min()
    x = (x - minm) / (maxm - minm)
    return x

path = "H:/networks\pl-models\miniset\latest/reference/1.jpg"
img = Image.open(path).convert('RGB')
img = transforms.Resize((224, 224))(img)
img = transforms.ToTensor()(img)
x = transforms.Normalize(OPENAI_MEAN, OPENAI_STD)(img).unsqueeze(0)
text = "the girl's blue eyes"

img = cv2.imread(path)
height, width = img.shape[:2]

clip = OpenCLIP(type="tokens").cuda()
# clip = OpenCLIP(arch="ViT-bigG-14", type="tokens", device="cpu")

v = clip.encode(x)
t = clip.encode_text(text)
# v = v / v.norm(dim=2, keepdim=True)
scale = clip.calculate_scale(v, t)
scale = scale.permute(0, 2, 1).view(1, 1, 16, 16)
scale = torch.nn.functional.interpolate(scale, size=(height, width), mode="bicubic").squeeze(0).view(1, height*width)
scale = maxmin(scale)
scale = scale.view(1, height, width).permute(1, 2, 0).cpu().numpy()
scale = (scale * 255.).astype(np.uint8)
heatmap = cv2.applyColorMap(scale, cv2.COLORMAP_JET)

result = cv2.addWeighted(img, 0.3, heatmap, 0.7, 0)

cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("heatmap", result)
cv2.waitKey()
