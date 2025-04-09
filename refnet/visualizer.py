import cv2
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn

import xformers
from refnet.util import default, exists


def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class CrossAttnVisualizer:
    token_length = 16
    reference = 0
    sketch = 1
    def hack(self, sd, vh, vw, target=0):
        """

        Args:
            sd: Denoising U-Net.
            block_id: Transformer index.
            vh: Horizontal index of visualization target region.
            vw: Vertical index of visualization target region.

        """
        def hack_crossattn_forward(self, x, context=None, mask=None, scale=1.):
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            if outer.target == outer.sketch:
                attention_map = torch.softmax(torch.bmm(q, k.permute(0, 2, 1))[:, outer.vid: outer.vid + 1], dim=2)
            else:
                attention_map = torch.softmax(torch.bmm(q, k.permute(0, 2, 1))[:, :, outer.vid: outer.vid + 1], dim=1)
            outer.attention_map_per_head.append(attention_map.reshape(
                b, self.heads, int(outer.token_length**2)
            ).contiguous().detach().float().cpu()[:1])

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
            out = out * scale

            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            return self.to_out(out)

        outer = self
        self.target = target
        for idx in sd.attn_modules["low"]:
            transformer = sd.attn_modules["modules"][idx]
            transformer.attn2.forward = hack_crossattn_forward.__get__(transformer.attn2, transformer.attn2.__class__)
        self.vid = vh * self.token_length + vw
        self.attention_map_per_head = None

    @property
    def avg_topk_attention_map(self):
        # Only visualize the average of [top_k] heads
        # Attention map shape: [5, heads, clip_token_number]
        # Five is the number of low-level cross-attention
        return self.attention_map_per_head.permute(1, 0, 2).reshape(-1, self.token_length, self.token_length).contiguous()

    def visualize_attention_map(self, reference):
        h, w, c = reference.shape

        attnmaps = (self.avg_topk_attention_map.numpy() * 255.).astype(np.uint8)
        weighted = []
        for idx, attnmap in enumerate(attnmaps):
            attnmap = np.expand_dims(attnmap, 0).transpose(1, 2, 0)
            heatmap = cv2.cvtColor(cv2.applyColorMap(attnmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
            weighted.append(cv2.addWeighted(reference, 0.3, heatmap, 0.7, 0))
        weighted = self.split_regions(weighted)
        return [Image.fromarray(r) for r in weighted]

    def split_regions(self, img):
        img = np.array(img)
        h, w, c = img.shape
        hu, wu = h // self.token_length, w // self.token_length

        for i in range(self.token_length):
            img[i * hu, :] = (0, 0, 0)
        for i in range(self.token_length):
            img[:, i * wu] = (0, 0, 0)
        return img

    def highlight(self, sketch, vh, vw):
        hu, wu = vh // self.token_length, vw // self.token_length
        heatmap = sketch
        heatmap[vh*hu: (vh+1)*hu, vw*wu: (vw+1)*wu] = (0, 0, 128)
        weighted = cv2.addWeighted(sketch, 0.5, heatmap, 0.5, 0)
        return [Image.fromarray(weighted.astype(np.uint8))]