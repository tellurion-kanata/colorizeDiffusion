from .sampling import *


def create_noise_sampler(x, sigmas, seed):
    """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
    from k_diffusion.sampling import BrownianTreeNoiseSampler
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)