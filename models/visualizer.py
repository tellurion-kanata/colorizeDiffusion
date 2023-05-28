import cv2
import torch
import torch.nn.functional as F
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

def maxmin2(s: torch.Tensor, threshold=0.5):
    """
        The shape of input scales tensor should be (b, n, 1)
    """
    assert len(s.shape) == 3
    ms = gap(s)
    maxm = s.max(dim=1, keepdim=True).values
    minm = s.min(dim=1, keepdim=True).values
    d = maxm - minm

    corr_s = (s - minm) / d
    corr_mean = (ms - minm) / d
    return torch.where(corr_s - corr_mean > 0, torch.exp(torch.abs(s-ms) * 0.5), -torch.exp(torch.abs(s-ms)))

def normalize(path):
    img = Image.open(path).convert('RGB')
    img = transforms.Resize((224, 224))(img)
    img = transforms.ToTensor()(img)
    x = transforms.Normalize(OPENAI_MEAN, OPENAI_STD)(img).unsqueeze(0)
    return x

# path = "H:/networks/pl-models/miniset/origin/82411909.png"
path = "H:/networks\pl-models\miniset\mapping/reference/1.jpg"
# path1 = "H:/networks/pl-models/generated/2.png"
# path2 = "H:/networks/pl-models/generated/0.png"
# text = ["hair", "red hair"]
text = ["white shirt"]
# x = torch.cat([normalize(path1), normalize(path2)], 0)
x = normalize(path)
# text = ["the girl's hair"]

img = cv2.imread(path)
height, width = img.shape[:2]

clip = OpenCLIP(type="full", layer="penultimate").cuda()
# clip = OpenCLIP(arch="ViT-bigG-14", type="tokens", device="cpu")

v = clip.encode(x)
t = clip.encode_text(text)
# t = v[:, 0].unsqueeze(1)
print(v.max())
v = v[:, 1:]
scale = clip.calculate_scale(v, t)
print(scale.view(-1, 16, 16, 1))
# gap_scale = gap(scale)
# scale = scale - gap_scale
# scale = torch.where(scale > 0, torch.ones_like(scale), torch.zeros_like(scale))
# scale = scale[0] - scale[1]
# print(scale.shape)
# scale = scale.unsqueeze(0)
# scale = torch.where(scale - gapdscale > 0, torch.ones_like(scale), torch.zeros_like(scale))

scale = scale.permute(0, 2, 1).view(1, 1, 16, 16)
scale = torch.nn.functional.interpolate(scale, size=(height, width), mode="bicubic").squeeze(0).view(1, height*width)
scale = maxmin(scale)

# scale = maxmin(F.softmax(-scale, dim=-1))
scale = scale.view(1, height, width).permute(1, 2, 0).cpu().numpy()
scale = (scale * 255.).astype(np.uint8)
heatmap = cv2.applyColorMap(scale, cv2.COLORMAP_JET)

result = cv2.addWeighted(img, 0.3, heatmap, 0.7, 0)

cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("heatmap", result)
cv2.waitKey()