import torch
import torch.nn as nn

def gap(x: torch.Tensor = None, keepdim=True):
    if len(x.shape) == 4:
        return torch.mean(x, dim=[2, 3], keepdim=keepdim)
    elif len(x.shape) == 3:
        return torch.mean(x, dim=[1], keepdim=keepdim)
    else:
        raise NotImplementedError('gap input should be 3d or 4d tensors')

def sine_loss(fv, ov, text_features, text_norm):
    """
        Compute the loss along the sine direction.
        fv: fake visual features generated by mapper network
        ov: original visual features
    """
    fv, ov = map(lambda t: t/t.norm(dim=2, keepdim=True), (fv, ov))
    text_features = text_features.permute(0, 2, 1)

    f_proj, o_proj = map(lambda t: (torch.bmm(t, text_features)) ** 2. / text_norm, (fv, ov))
    fv, ov = map(lambda t: (t ** 2).sum(dim=2, keepdim=True), (fv, ov))

    sin_fv = fv - f_proj
    sin_ov = ov - o_proj
    return torch.abs(sin_ov - sin_fv)


class MappingLoss(nn.Module):
    def __init__(self, trigular_weight=100.0):
        super().__init__()
        self.trigular_weight = trigular_weight

    def forward(self, x, sketch, fake_crossattn, real_crossattn, origin_crossattn, text_features,
                calculate_scale, diffusion_model, split="train"):
        b, n, c = text_features.shape

        t = torch.randint(0, diffusion_model.num_timesteps, (x.shape[0],), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = diffusion_model.q_sample(x_start=x, t=t, noise=noise)

        inv_loss = (torch.abs(gap(real_crossattn) - gap(fake_crossattn))).sum() / b

        fake = diffusion_model.apply_model(x_noisy, t, {"c_concat": [sketch], "c_crossattn": [fake_crossattn]})
        real = diffusion_model.apply_model(x_noisy, t, {"c_concat": [sketch], "c_crossattn": [real_crossattn]})
        nll_loss = (torch.abs(real - fake)).sum() / b

        # MSE-triangular loss
        real_scale = calculate_scale(real_crossattn, text_features)
        fake_scale = calculate_scale(fake_crossattn, text_features)
        text_features = text_features / text_features.norm(dim=2, keepdim=True)
        text_norm = (text_features ** 2).sum(dim=2).unsqueeze(1)

        cos_loss = torch.abs((text_norm * real_scale) ** 2. - (text_norm * fake_scale) ** 2.)
        sin_loss = sine_loss(fake_crossattn, origin_crossattn, text_features, text_norm)
        tri_loss = self.trigular_weight * (cos_loss + sin_loss).sum() / (b * n)

        loss = nll_loss + tri_loss + inv_loss
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/inv_loss".format(split): inv_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/tri_loss".format(split): tri_loss.detach().mean(),
               "{}/sin_loss".format(split): sin_loss.detach().mean(),
               "{}/cos_loss".format(split): cos_loss.detach().mean(),}
        return loss, log