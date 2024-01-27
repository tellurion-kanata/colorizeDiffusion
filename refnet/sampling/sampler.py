import dataclasses
import torch
import k_diffusion
import inspect

from types import SimpleNamespace
from refnet.util import default
from .scheduler import schedulers, schedulers_map
from .denoiser import CFGDenoiser

defaults = SimpleNamespace(**{
    "eta_ddim": 0.0,
    "eta_ancestral": 1.0,
    "ddim_discretize": "uniform",
    "s_churn": 0.0,
    "s_tmin": 0.0,
    "s_noise": 1.0,
    "k_sched_type": "Automatic",
    "sigma_min": 0.0,
    "sigma_max": 0.0,
    "rho": 0.0,
    "eta_noise_seed_delta": 0,
    "always_discard_next_to_last_sigma": False,
})

@dataclasses.dataclass
class Sampler:
    label: str
    funcname: str
    aliases: any
    options: dict


samplers_k_diffusion = [
    Sampler('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {'scheduler': 'karras'}),
    Sampler('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    Sampler('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde'], {'scheduler': 'exponential', "brownian_noise": True}),
    Sampler('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
    Sampler('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    Sampler('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    Sampler('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    Sampler('Euler', 'sample_euler', ['k_euler'], {}),
    Sampler('LMS', 'sample_lms', ['k_lms'], {}),
    Sampler('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    Sampler('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "second_order": True}),
    Sampler('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    Sampler('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    Sampler('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True})
]

sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_fast': ['s_noise'],
    'sample_dpm_2_ancestral': ['s_noise'],
    'sample_dpmpp_2s_ancestral': ['s_noise'],
    'sample_dpmpp_sde': ['s_noise'],
    'sample_dpmpp_2m_sde': ['s_noise'],
    'sample_dpmpp_3m_sde': ['s_noise'],
}

def kdiffusion_sampler_list():
    return [k.label for k in samplers_k_diffusion]


k_diffusion_samplers_map = {x.label: x for x in samplers_k_diffusion}
k_diffusion_scheduler = {x.name: x.function for x in schedulers}

def exists(v):
    return v is not None


class KDiffusionSampler:
    def __init__(self, sampler, scheduler, sd, device):
        # k_diffusion_samplers_map[]
        self.config = k_diffusion_samplers_map[sampler]
        funcname = self.config.funcname

        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, funcname)
        self.scheduler_name = scheduler
        self.sd = CFGDenoiser(sd, device)
        self.model_wrap = self.sd.model_wrap
        self.device = device

        self.s_min_uncond = None
        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')
        self.s_noise = 1.0

        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0
        self.eta = None

        self.extra_params = []

        if exists(sd.sigma_max) and exists(sd.sigma_min):
            self.model_wrap.sigmas[-1] = sd.sigma_max
            self.model_wrap.sigmas[0] = sd.sigma_min

    def initialize(self):
        self.eta = getattr(defaults, self.eta_option_field, 0.0)

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(self, param_name)

        if 'eta' in inspect.signature(self.func).parameters:
            extra_params_kwargs['eta'] = self.eta

        if len(self.extra_params) > 0:
            s_churn = getattr(defaults, 's_churn', self.s_churn)
            s_tmin = getattr(defaults, 's_tmin', self.s_tmin)
            s_tmax = getattr(defaults, 's_tmax', self.s_tmax) or self.s_tmax  # 0 = inf
            s_noise = getattr(defaults, 's_noise', self.s_noise)

            if 's_churn' in extra_params_kwargs and s_churn != self.s_churn:
                extra_params_kwargs['s_churn'] = s_churn
                self.s_churn = s_churn
            if 's_tmin' in extra_params_kwargs and s_tmin != self.s_tmin:
                extra_params_kwargs['s_tmin'] = s_tmin
                self.s_tmin = s_tmin
            if 's_tmax' in extra_params_kwargs and s_tmax != self.s_tmax:
                extra_params_kwargs['s_tmax'] = s_tmax
                self.s_tmax = s_tmax
            if 's_noise' in extra_params_kwargs and s_noise != self.s_noise:
                extra_params_kwargs['s_noise'] = s_noise
                self.s_noise = s_noise

        return extra_params_kwargs

    def create_noise_sampler(self, x, sigmas, seed):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        from k_diffusion.sampling import BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)

    def get_sigmas(self, steps, sigmas_min=None, sigmas_max=None):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)

        steps += 1 if discard_next_to_last_sigma else 0

        if self.scheduler_name == 'Automatic':
            self.scheduler_name = self.config.options.get('scheduler', None)

        scheduler = schedulers_map.get(self.scheduler_name)
        sigma_min = default(sigmas_min, self.model_wrap.sigma_min)
        sigma_max = default(sigmas_max, self.model_wrap.sigma_max)

        if scheduler is None or scheduler.function is None:
            sigmas = self.model_wrap.get_sigmas(steps)
        else:
            sigmas_kwargs = {'sigma_min': sigma_min, 'sigma_max': sigma_max}

            if scheduler.need_inner_model:
                sigmas_kwargs['inner_model'] = self.model_wrap

            sigmas = scheduler.function(n=steps, **sigmas_kwargs, device=self.device)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas


    def __call__(self, x, sigmas, sampler_extra_args, seed, deterministic, steps=None):
        x = x * sigmas[0]

        extra_params_kwargs = self.initialize()
        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters:
            extra_params_kwargs['n'] = steps

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = sigmas[sigmas > 0].min()
            extra_params_kwargs['sigma_max'] = sigmas.max()

        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigmas

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, seed) if deterministic else None
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        return self.func(self.sd, x, extra_args=sampler_extra_args, disable=False, **extra_params_kwargs)
