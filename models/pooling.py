import torch
import torch.nn as nn
import numpy.random as random


class LatentShuffle(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x:torch.Tensor, sample=False):
        if self.training or sample:
            b, n, c = x.shape

            # shuffle the latent code
            new_ind = torch.randperm(n)
            shuffled = x[:, new_ind]

            # return shuffled latent code
            return shuffled.contiguous()
        return x


class LatentPooling(nn.Module):
    init_methods = {
        "zeros": torch.zeros,
        "randn": torch.randn
    }

    def __init__(self,
                 p=0.2,
                 pool_size=2560,
                 dim=1024,
                 init="zeros",
                 ):
        super().__init__()
        assert init in ["zeros", "randn"]
        initialize = self.init_methods[init]

        self.p = p
        self.pool_size = pool_size
        self.latent_code_pool = initialize([pool_size, dim], requires_grad=False)

    def forward(self, x:torch.Tensor, sample=False):
        if self.training or sample:
            b, n, c = x.shape

            # shuffle the latent code
            new_ind = torch.randperm(n)
            replaced = x[:, new_ind]

            # replace n0 latent codes, n0 is randomly selected in [0, n)
            replace_num = int(random.rand() * self.p * n)
            replace_ind = torch.randperm(n)[:replace_num]
            noise_ind = torch.randperm(self.pool_size)[:replace_num]
            update_codes = replaced[0][replace_ind]
            replaced[:, replace_ind] = self.latent_code_pool[noise_ind].to(x.dtype).to(x.device)

            # update latent code pool
            self.latent_code_pool[noise_ind] = update_codes.data.to(self.latent_code_pool.dtype).to(self.latent_code_pool.device)

            # return shuffled and replaced latent code
            return replaced.contiguous().detach()
        return x