import os
import random
import traceback
import gradio as gr
import os.path as osp

from datetime import datetime
from glob import glob

from util import load_config
from refnet.util import instantiate_from_config, scaled_resize
from refnet.visualizer import CrossAttnVisualizer
from preprocessor import create_model
from .functool import *


model = None
visualizer = CrossAttnVisualizer()

model_type = ""
model_path = "models"
current_checkpoint = ""
attn_vispath = "visualization"
default_line_extractor = "lineart_keras"
default_mask_extractor = "rmbg-v2"
global_seed = None

smask_extractor = create_model("ISNet-sketch").cpu()

MAXM_INT32 = 429496729
model_types = ["text", "mult", "switch", "old", "v2", "sdxl", "cluster", "logit", "mapper", "xlv2"]



'''
    Gradio UI functions
'''


def switch_to_fp16():
    global model
    model.switch_to_fp16()
    gr.Info("Switch unet to half precision")


def switch_vae_to_fp16():
    global model_type
    model.switch_vae_to_fp16()
    gr.Info("Switch vae to half precision")


def switch_to_fp32():
    global model
    model.switch_to_fp32()
    gr.Info("Switch unet to full precision")


def switch_vae_to_fp32():
    global model_type
    model.switch_vae_to_fp32()
    gr.Info("Switch vae to full precision")


def get_checkpoints():
    ckpt_fmts = ["safetensors", "pth", "ckpt", "pt"]
    checkpoints = sorted([
        osp.basename(file) for ext in ckpt_fmts
        for file in glob(osp.join(model_path, f"*.{ext}"))
    ])
    return checkpoints


def update_models():
    global current_checkpoint
    checkpoints = get_checkpoints()
    if not checkpoints:
        return gr.update(choices=[], value=None)
    if current_checkpoint not in checkpoints:
        current_checkpoint = checkpoints[0]
    return gr.update(choices=checkpoints, value=current_checkpoint)


def switch_extractor(type):
    global line_extractor
    try:
        line_extractor = create_model(type)
        gr.Info(f"Switched to {type} extractor")
    except:
        gr.Info(f"Failed in loading {type} extractor")


def switch_mask_extractor(type):
    global mask_extractor
    if type != "none":
        try:
            mask_extractor = create_model(type)
            gr.Info(f"Switched to {type} extractor")
        except:
            gr.Info(f"Failed in loading {type} extractor")


def apppend_prompt(target, anchor, control, scale, enhance, ts0, ts1, ts2, ts3, prompt):
    target = target.strip()
    anchor = anchor.strip()
    control = control.strip()
    if target == "": target = "none"
    if anchor == "": anchor = "none"
    if control == "": control = "none"
    new_p = (f"\n[target] {target}; [anchor] {anchor}; [control] {control}; [scale] {str(scale)}; "
             f"[enhanced] {str(enhance)}; [ts0] {str(ts0)}; [ts1] {str(ts1)}; [ts2] {str(ts2)}; [ts3] {str(ts3)}")
    return "", "", "", 0.0, False, 0.5, 0.55, 0.65, 0.95, (prompt + new_p).strip()


def clear_prompts():
    return ""


def load_model(ckpt_path):
    global model, model_type, current_checkpoint
    config_root = "configs/inference"

    try:
        new_model_type = model_type
        for key in model_types:
            if ckpt_path.startswith(key):
                new_model_type = key
                break

        if model_type != new_model_type or not "model" in globals():
            if "model" in globals() and exists(model):
                del model
            config_path = osp.join(config_root, f"{new_model_type}.yaml")
            new_model = instantiate_from_config(load_config(config_path).model).cpu().eval()
            print(f"Swithced to {model_type} model, loading weights from [{ckpt_path}]...")
            model = new_model

        model.parameterization = "eps" if ckpt_path.find("eps") > -1 else "v"
        model.init_from_ckpt(osp.join(model_path, ckpt_path), logging=True)
        model.switch_to_fp16()

        model_type = new_model_type
        current_checkpoint = ckpt_path
        print(f"Loaded model from [{ckpt_path}], model_type [{model_type}].")
        gr.Info("Loaded model successfully.")

    except Exception as e:
        print(f"Error type: {e}")
        print(traceback.print_exc())
        gr.Info("Failed in loading model.")


def get_last_seed():
    return global_seed or -1


def reset_random_seed():
    return -1


def visualize(reference, text, *args):
    return visualize_heatmaps(model, reference, parse_prompts(text), *args)

def split_sketch(image, vh, vw):
    splited = visualizer.split_regions(
        resize((512, 512))(
            pad_image_with_margin(image, 1)
        )
    )
    return visualizer.highlight(splited, vh, vw)


def visualize_attention_map(reference, result, vh, vw):
    crossattn_heatmaps = visualizer.visualize_attention_map(reference)
    splited_result = visualizer.split_regions(result)
    splited_result = visualizer.highlight(splited_result, vh, vw)

    current_time = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    save_path = osp.join(attn_vispath, f"{current_time}")
    os.makedirs(save_path, exist_ok=True)

    for idx, casheat in enumerate(crossattn_heatmaps):
        casheat.save(osp.join(save_path, f"head-{idx}_vh-{vh}_vw-{vw}.png"))
    splited_result[0].save(osp.join(save_path, f"sketch_vh-{vh}_vw-{vw}.png"))

    return splited_result

def set_cas_scales(accurate, cas_args):
    enc_scale, middle_scale, low_scale, strength = cas_args[:4]
    attn_scales = cas_args[5:]
    if not accurate:
        scale_strength = {
            "level_control": True,
            "scales": {
                "encoder": enc_scale * strength,
                "middle": middle_scale * strength,
                "low": low_scale * strength,
            }
        }
    else:
        scale_strength = {
            "level_control": False,
            "scales": list(attn_scales)
        }
    return scale_strength


@torch.no_grad()
def inference(
        style_enhance, bg_enhance, fg_enhance, fg_disentangle_scale,
        bs, input_s, input_r, input_bg, mask_ts, mask_ss, gs_r, gs_s, ctl_scale,
        fg_strength, bg_strength, merge_scale, mask_scale, height, width, seed, low_vram, step,
        injection, remove_fg, rmbg, infid_x, infid_r, injstep, crop, pad_scale,
        start_step, end_step, no_start_step, no_end_step, return_inter, sampler, scheduler, preprocess,
        deterministic, text, target, anchor, control, target_scale, ts0, ts1, ts2, ts3, enhance, accurate,
        *args
):
    global global_seed, line_extractor, mask_extractor
    global_seed = seed if seed > -1 else random.randint(0, MAXM_INT32)
    torch.manual_seed(global_seed)

    # if vis_crossattn:
    #     assert (height, width) == (512, 512), "Only to visualize standard results at 16 latent scale"
    #     visualizer.hack(model, vh, vw, width)
    smask, rmask, bgmask = None, None, None
    # if exists(input_s):
    #     input_s, smask = input_s["image"], input_s["mask"]
    # if exists(input_r):
    #     input_r, rmask = input_r["image"], input_r["mask"]
    # if exists(input_bg):
    #     input_bg, bgmask = input_bg["image"], input_bg["mask"]

    manipulation_params = parse_prompts(text, target, anchor, control, target_scale, ts0, ts1, ts2, ts3, enhance)
    inputs = preprocessing_inputs(
        sketch = input_s,
        reference = input_r,
        background = input_bg,
        preprocess = preprocess,
        hook = injection,
        resolution = (height, width),
        extractor = line_extractor,
        pad_scale = pad_scale,
    )
    sketch, reference, background, original_shape, inject_xr, inject_xs, white_sketch = inputs
    if not osp.exists(attn_vispath):
        os.makedirs(attn_vispath)

    cond = {"reference": reference, "sketch": sketch, "background": background}
    mask_guided = bg_enhance or fg_enhance

    if exists(white_sketch) and exists(reference) and mask_guided:
        mask_extractor.cuda()
        smask_extractor.cuda()
        smask = smask_extractor.proceed(x=white_sketch, pil_x=input_s, th=height, tw=width, threshold=mask_ss, crop=False)

        if exists(background) and remove_fg:
            bgmask = mask_extractor.proceed(x=background, pil_x=input_bg, threshold=mask_ts, dilate=True)
            filtered_background = torch.where(bgmask < mask_ts, background, torch.ones_like(background))
            cond.update({"background": filtered_background, "rmask": bgmask})
            
        else:
            rmask = mask_extractor.proceed(x=reference, pil_x=input_r, threshold=mask_ts, dilate=True)
            cond.update({"rmask": rmask})

        cond.update({"smask": scaled_resize(smask, 0.125)})
        smask_extractor.cpu()
        mask_extractor.cpu()

    # if hasattr(model.cond_stage_model, "scale_factor") and scale_factor != model.cond_stage_model.scale_factor:
    #     model.cond_stage_model.update_scale_factor(scale_factor)
    scale_strength = set_cas_scales(accurate, args)

    results = model.generate(
        # Colorization mode
        style_enhance = style_enhance,
        bg_enhance = bg_enhance,
        fg_enhance = fg_enhance,
        fg_disentangle_scale = fg_disentangle_scale,

        # Conditional inputs
        cond = cond,
        ctl_scale = ctl_scale,
        merge_scale = merge_scale,
        mask_scale = mask_scale,
        mask_thresh = mask_ts,
        mask_thresh_sketch = mask_ss,

        # Sampling settings
        bs = bs,
        gs = [gs_r, gs_s],
        sampler = sampler,
        scheduler = scheduler,
        start_step = start_step,
        end_step = end_step,
        no_start_step = no_start_step,
        no_end_step = no_end_step,
        strength = scale_strength,
        fg_strength = fg_strength,
        bg_strength = bg_strength,
        seed = global_seed,
        deterministic = deterministic,
        height = height,
        width = width,
        step = step,

        # Injection settings
        injection = injection,
        injection_cfg = infid_r,
        injection_control = infid_x,
        injection_start_step = injstep,
        hook_xr = inject_xr,
        # hook_xs = inject_xs,

        # Additional settings
        low_vram = low_vram,
        return_intermediate = return_inter,
        manipulation_params = manipulation_params,
    )

    if rmbg:
        mask_extractor.cuda()
        mask = mask_extractor.proceed(x=results, threshold=mask_ts)
        results = torch.where(mask >= mask_ts, results, torch.ones_like(results))
        mask_extractor.cpu()

    results = postprocess(results, sketch, reference, background, crop, original_shape,
                          mask_guided, smask, rmask, bgmask, mask_ts, mask_ss)
    torch.cuda.empty_cache()
    gr.Info("Generation completed.")
    return results
