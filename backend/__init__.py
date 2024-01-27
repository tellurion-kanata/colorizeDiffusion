from .appfunc import *


__all__ = [
    'switch_extractor', 'switch_mask_extractor', 'switch_to_fp16', 'switch_to_fp32', 'split_sketch',
    'get_checkpoints', 'load_model', 'inference', 'update_models', 'reset_random_seed', 'get_last_seed',
    'switch_vae_to_fp16', 'switch_vae_to_fp32', 'apppend_prompt', 'clear_prompts', 'visualize',
    'default_line_extractor', 'default_mask_extractor', 'MAXM_INT32', 'mask_extractor_list', 'line_extractor_list'
]


mask_extractor_list = ["ISNet", "rmbg-v2"]
line_extractor_list = ["lineart", "lineart_denoise", "lineart_keras", "lineart_sk"]