import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

from ldm.modules.encoders.modules import OpenCLIP, OPENAI_MEAN, OPENAI_STD


def maxmin(x: torch.Tensor):
    maxm = x.max()
    minm = x.min()
    x = (x - minm) / (maxm - minm)
    return x

path = "../miniset/origin/79265251.png"
img = Image.open(path).convert('RGB')
img = transforms.Resize((224, 224))(img)
img = transforms.ToTensor()(img)
x = transforms.Normalize(OPENAI_MEAN, OPENAI_STD)(img).unsqueeze(0)
text = "hair"

img = cv2.imread(path)
height, width = img.shape[:2]

# clip = OpenCLIP(type="tokens").cuda()
clip = OpenCLIP(arch="ViT-bigG-14", type="tokens", device="cpu")

v = clip.encode_image(x)
t = clip.encode_text(text)
scale = clip.calculate_scale(v, t).permute(0, 2, 1).view(1, 1, 16, 16)
scale = torch.nn.functional.interpolate(scale, size=(height, width), mode="bicubic").squeeze(0).view(1, height*width)
scale = maxmin(scale).view(1, height, width).permute(1, 2, 0).cpu().numpy()
scale = (scale * 255.).astype(np.uint8)
heatmap = cv2.applyColorMap(scale, cv2.COLORMAP_JET)

result = cv2.addWeighted(img, 0.3, heatmap, 0.7, 0)

cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
cv2.imshow("heatmap", result)
cv2.waitKey()
