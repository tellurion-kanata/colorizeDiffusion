import sys
import random
import gradio as gr
import os.path as osp

from glob import glob
from util import load_config
from sgm.util import instantiate_from_config
from refnet.models.basemodel import get_sampler_list
from preprocessor import create_model
from ui_backend.functool import *


model = None
model_type = ""
maxium_resolution = 2048
model_path = "models"
MAXM_INT32 = 4294967295

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
    checkpoints = [
        osp.basename(file) for ext in ckpt_fmts
        for file in glob(osp.join(model_path, f"*.{ext}"))
    ]
    return checkpoints

def update_models():
    return gr.update(choices=get_checkpoints(), value=None)


def switch_extractor(type):
    global extractor
    try:
        extractor = create_model(type)
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
    model_types = {
        "proj": "proj",
        "rot": "rot",
        "hybrid": "hybrid",
        "vanilla": "colorizer",
        "shuffle": "colorizer",
        "deform": "colorizer",
    }
    global model, model_type
    config_root = "configs/inference"

    try:
        new_model_type = model_type

        for key in model_types.keys():
            if ckpt_path.find(key) > -1:
                new_model_type = model_types[key]

        if model_type != new_model_type:
            if exists(model):
                del model
            model_type = new_model_type
            config_path = osp.join(config_root, f"{model_type}.yaml")
            new_model = instantiate_from_config(load_config(config_path).model).cpu().eval()
            print(f"Swithced to {model_type} model")
            model = new_model

        model.init_from_ckpt(osp.join(model_path, ckpt_path))
        model.switch_to_fp16()
        print(f"Loaded model from [{ckpt_path}], model_type [{model_type}].")
        gr.Info("Loaded model successfully.")

    except:
        gr.Info("Failed in loading model.")


def get_last_seed():
    return global_seed


def reset_random_seed():
    return -1


def visualize(reference, text, *args):
    return visualize_heatmaps(model, reference, parse_prompts(text), *args)

def inference(bs, sketch, reference, gs_r, gs_s, resolution, seed, low_vram, step, ref_noise,
              injection, injection_fidelity, adain, adain_fidelity, crop, noise_level, sampler, preprocess,
              scale_factor, use_local, text, *args):

    global global_seed, extractor
    torch.cuda.empty_cache()
    global_seed = seed if seed > -1 else random.randint(0, MAXM_INT32)
    torch.manual_seed(global_seed)

    inputs = preprocessing_inputs(sketch, reference, preprocess, injection or adain, resolution, extractor, ref_noise)
    sketch, reference, original_shape, inject_sketch, inject_xr = inputs

    if hasattr(model.cond_stage_model, "scale_factor") and scale_factor != model.cond_stage_model.scale_factor:
        model.cond_stage_model.update_scale_factor(scale_factor)

    results = model.generate(
        bs = bs,
        cond = {"crossattn": reference, "concat": sketch},
        gs = [gs_r, gs_s],
        height = resolution,
        width = resolution,
        step = step,
        injection = injection,
        injection_cfg = injection_fidelity,
        adain = adain,
        gn_weight = adain_fidelity,
        noise_level = noise_level,
        low_vram = low_vram,
        sampler = sampler,
        hook_xr = inject_xr,
        hook_sketch = inject_sketch,
        use_local = use_local,
        use_rx = ref_noise,
        manipulation_params = parse_prompts(text, *args),
    )
    results = to_numpy(results)
    sketch = to_numpy(sketch)[0]

    results_list = []
    for result in results:
        result = Image.fromarray(result)
        if crop:
            result = crop_image_from_square(result, original_shape)
        results_list.append(result)
    results_list.append(sketch)
    return results_list


def init_inerface() -> None:
    sampler_list = get_sampler_list()
    with gr.Blocks(title="Colorize Diffusion", css="div.gradio-container{ max-width: unset !important; }") as block:
        with gr.Row():
            gr.Markdown(f"## Colorize Diffusion")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    target = gr.Textbox(label="Target prompt", value="", scale=2)
                    anchor = gr.Textbox(label="Anchor prompt", value="", scale=2)
                    control = gr.Textbox(label="Control prompt", value="", scale=2)
                with gr.Row():
                    target_scale = gr.Slider(label="Target scale", value=0.0, minimum=0, maximum=15.0, step=0.25, scale=2)
                    ts0 = gr.Slider(label="Threshold 0", value=0.5, minimum=0, maximum=1.0, step=0.01)
                    ts1 = gr.Slider(label="Threshold 1", value=0.55, minimum=0, maximum=1.0, step=0.01)
                    ts2 = gr.Slider(label="Threshold 2", value=0.65, minimum=0, maximum=1.0, step=0.01)
                    ts3 = gr.Slider(label="Threshold 3", value=0.95, minimum=0, maximum=1.0, step=0.01)
                with gr.Row():
                    enhance = gr.Checkbox(label="Enhance manipulation", value=False)
                    add_prompt = gr.Button(value="Add")
                    clear_prompt = gr.Button(value="Clear")
                    vis_button = gr.Button(value="Visualize")
                text_prompt = gr.Textbox(label="Final prompt", value="", lines=3)

                with gr.Row():
                    sketch_img = gr.Image(label="Sketch", type="pil", height=256)
                    reference_img = gr.Image(label="Reference", type="pil", height=256)

                with gr.Row():
                    step = gr.Slider(label="Step", minimum=1, maximum=200, value=20, step=1, scale=2)
                    gs_r = gr.Slider(label="Reference Guidance Scale", minimum=1, maximum=15.0, value=5.0, step=0.5)
                    gs_s = gr.Slider(label="Sketch Guidance Scale", minimum=1, maximum=5.0, value=1.0, step=0.1)


                with gr.Accordion("Advanced Setting", open=True):
                    with gr.Row():
                        ref_noise = gr.Checkbox(label="Use reference noise", value=False)
                        crop = gr.Checkbox(label="Crop result", value=False)
                        adain = gr.Checkbox(label="Use AdaIN", value=False)
                        injection = gr.Checkbox(label="Use attention injection", value=False)
                    with gr.Row():
                        noise_level = gr.Slider(label="Reference noise level", minimum=0, maximum=1, value=0, step=0.05)
                        scale_factor = gr.Slider(label="Reference resize scale", minimum=1, maximum=2, value=1, step=0.25)
                        adain_fidelity = gr.Slider(label="AdaIN fidelity", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                        injection_fidelity = gr.Slider(label="Injection fidelity", minimum=0.0, maximum=1.0, value=0.5, step=0.05)

            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True)
                run_button = gr.Button(value="Run")
                with gr.Row():
                    vae_fp16 = gr.Button(value="fp16 vae")
                    vae_fp32 = gr.Button(value="fp32 vae")
                    fp16 = gr.Button(value="fp16 unet")
                    fp32 = gr.Button(value="fp32 unet")

                with gr.Row():
                    sd_model = gr.Dropdown(choices=get_checkpoints(), label="stable diffusion model",
                                           value=get_checkpoints()[0], scale=2)
                    sampler = gr.Dropdown(choices=sampler_list, value=sampler_list[-1], label="sampler", scale=2)
                    preprocessor = gr.Dropdown(choices=["none", "extract", "invert"], label="sketch preprocessor",
                                               value="invert")
                    extractor_model = gr.Dropdown(choices=["lineart", "lineart_denoise", "lineart_keras"], label="line extractor",
                                                  value="lineart")


                with gr.Row():
                    bs = gr.Slider(label="Batch size", minimum=1, maximum=8, value=1, step=1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=MAXM_INT32, step=1, value=-1)
                    resolution = gr.Slider(label="Resolution", minimum=256, maximum=2048, value=512, step=64, scale=2)

                with gr.Row():
                    reuse_seed = gr.Button(value="Reuse seed")
                    random_seed = gr.Button(value="Random seed")
                    update_ckpts = gr.Button(value="Update checkpoints")

                with gr.Row():
                    use_local = gr.Checkbox(label="Local token", value=False)
                    save_memory = gr.Checkbox(label="Save memory", value=True)

        add_prompt.click(fn=apppend_prompt,
                         inputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt],
                         outputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt])
        clear_prompt.click(fn=clear_prompts,
                           outputs=[text_prompt])
        sd_model.input(fn=load_model, inputs=[sd_model])
        extractor_model.input(fn=switch_extractor, inputs=[extractor_model])
        reuse_seed.click(fn=get_last_seed, outputs=[seed])
        random_seed.click(fn=reset_random_seed, outputs=[seed])
        update_ckpts.click(fn=update_models, outputs=[sd_model])

        fp16.click(fn=switch_to_fp16)
        fp32.click(fn=switch_to_fp32)
        vae_fp16.click(fn=switch_vae_to_fp16)
        vae_fp32.click(fn=switch_vae_to_fp32)

        ips = [bs, sketch_img, reference_img, gs_r, gs_s, resolution, seed, save_memory,
               step, ref_noise, injection, injection_fidelity, adain, adain_fidelity, crop,
               noise_level, sampler, preprocessor, scale_factor, use_local, text_prompt,
               target, anchor, control, target_scale, ts0, ts1, ts2, ts3, enhance]
        run_button.click(fn=inference, inputs=ips, outputs=[result_gallery])
        vis_button.click(fn=visualize,
                         inputs=[reference_img, text_prompt, control, ts0, ts1, ts2, ts3],
                         outputs=[result_gallery])

    block.launch(server_name="127.0.0.1", share=bool(sys.argv[1]))


if __name__ == '__main__':
    load_model(get_checkpoints()[0])
    switch_extractor("lineart")
    interface = init_inerface()