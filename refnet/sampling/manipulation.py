import torch
import torch.nn.functional as F

import numpy as np


def compute_pwv(s: torch.Tensor, dscale: torch.Tensor, ratio=2, thresholds=[0.5, 0.55, 0.65, 0.95]):
    """
        The shape of input scales tensor should be (b, n, 1)
    """
    assert len(s.shape) == 3, len(thresholds) == 4
    maxm = s.max(dim=1, keepdim=True).values
    minm = s.min(dim=1, keepdim=True).values
    d = maxm - minm

    maxmin = (s - minm) / d

    adjust_scale = torch.where(maxmin <= thresholds[0],
                               -dscale * ratio,
                               -dscale + dscale * (maxmin - thresholds[0]) / (thresholds[1] - thresholds[0]))
    adjust_scale = torch.where(maxmin > thresholds[1],
                               0.5 * dscale * (maxmin - thresholds[1]) / (thresholds[2] - thresholds[1]),
                               adjust_scale)
    adjust_scale = torch.where(maxmin > thresholds[2],
                               0.5 * dscale + 0.5 * dscale * (maxmin - thresholds[2]) / (thresholds[3] - thresholds[2]),
                               adjust_scale)
    adjust_scale = torch.where(maxmin > thresholds[3], dscale, adjust_scale)
    return adjust_scale


def local_manipulate_step(clip, v, t, target_scale, a=None, c=None, enhance=False, thresholds=[]):
    # print(f"target:{t}, anchor:{a}")
    cls_token = v[:, 0].unsqueeze(1)
    v = v[:, 1:]

    cur_target_scale = clip.calculate_scale(cls_token, t)
    # control_scale = clip.calculate_scale(cls_token, c)
    # print(f"current global target scale: {cur_target_scale},",
    #       f" global control scale: {control_scale}")

    if a is not None and a != "none":
        a = [a] * v.shape[0]
        a = clip.encode_text(a)
        anchor_scale = clip.calculate_scale(cls_token, a)
        dscale = target_scale - cur_target_scale if not enhance else target_scale - anchor_scale
        # print(f"global anchor scale: {anchor_scale}")

        c_map = clip.calculate_scale(v, c)
        a_map = clip.calculate_scale(v, a)
        pwm = compute_pwv(c_map, dscale, thresholds=thresholds) if c != "everything" else dscale
        base = 1 if enhance else 0
        v = v + (pwm + base * a_map) * (t - a)
    else:
        dscale = target_scale - cur_target_scale
        c_map = clip.calculate_scale(v, c)
        pwm = compute_pwv(c_map, dscale, thresholds=thresholds) if c != "everything" else dscale
        v = v + pwm * t
    v = torch.cat([cls_token, v], dim=1)
    return v

def local_manipulate(clip, v, targets, target_scales, anchors, controls, enhances=[], thresholds_list=[]):
    """
        v: visual tokens in shape (b, n, c)
        target: target text embeddings in shape (b, 1 ,c)
        control: control text embeddings in shape (b, 1, c)
    """
    controls, targets = clip.encode_text(controls + targets).chunk(2)
    for t, a, c, s_t, enhance, thresholds in zip(targets, anchors, controls, target_scales, enhances, thresholds_list):
        v = local_manipulate_step(clip, v, t, s_t, a, c, enhance, thresholds)
    return v


def global_manipulate_step(clip, v, t, target_scale, a=None, enhance=False):
    if a is not None and a != "none":
        a = [a] * v.shape[0]
        a = clip.encode_text(a)
        if enhance:
            s_a = clip.calculate_scale(v, a)
            v = v - s_a * a
        else:
            v = v + target_scale * (t - a)
            return v
    if enhance:
        v = v + target_scale * t
    else:
        cur_target_scale = clip.calculate_scale(v, t)
        v = v + (target_scale - cur_target_scale) * t
    return v


def global_manipulate(clip, v, targets, target_scales, anchors, enhances):
    targets = clip.encode_text(targets)
    for t, a, s_t, enhance in zip(targets, anchors, target_scales, enhances):
        v = global_manipulate_step(clip, v, t, s_t, a, enhance)
    return v


def assign_heatmap(s: torch.Tensor, threshold: float):
    """
        The shape of input scales tensor should be (b, n, 1)
    """
    maxm = s.max(dim=1, keepdim=True).values
    minm = s.min(dim=1, keepdim=True).values
    d = maxm - minm
    return torch.where((s - minm) / d < threshold, torch.zeros_like(s), torch.ones_like(s) * 0.25)


def get_heatmaps(model, reference, height, width, vis_c, ts0, ts1, ts2, ts3,
                 controls, targets, anchors, thresholds_list, target_scales, enhances):
    model.low_vram_shift("cond")
    clip = model.cond_stage_model

    v = clip.encode(reference, "full")
    if len(targets) > 0:
        controls, targets = clip.encode_text(controls + targets).chunk(2)
        inputs_iter = zip(controls, targets, anchors, target_scales, thresholds_list, enhances)
        for c, t, a, target_scale, thresholds, enhance in inputs_iter:
            # update image tokens
            v = local_manipulate_step(clip, v, t, target_scale, a, c, enhance, thresholds)
    token_length = v.shape[1] - 1
    grid_num = int(token_length ** 0.5)
    vis_c = clip.encode_text([vis_c])
    local_v = v[:, 1:]
    scale = clip.calculate_scale(local_v, vis_c)
    scale = scale.permute(0, 2, 1).view(1, 1, grid_num, grid_num)
    scale = F.interpolate(scale, size=(height, width), mode="bicubic").squeeze(0).view(1, height * width)

    # calculate heatmaps
    heatmaps = []
    for threshold in [ts0, ts1, ts2, ts3]:
        heatmap = assign_heatmap(scale, threshold=threshold)
        heatmap = heatmap.view(1, height, width).permute(1, 2, 0).cpu().numpy()
        heatmap = (heatmap * 255.).astype(np.uint8)
        heatmaps.append(heatmap)
    return heatmaps