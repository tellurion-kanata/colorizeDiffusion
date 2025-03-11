import gradio as gr
import argparse

from refnet.sampling import get_noise_schedulers, get_sampler_list
from backend import *

def app_optiosn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", '-addr', type=str, default="0.0.0.0")
    parser.add_argument("--server_port", '-port', type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--not_show_error", action="store_true")
    parser.add_argument("--enable_text_manipulation", '-manipulate', action="store_true")

    return parser.parse_args()


def init_inerface(opt, *args, **kwargs) -> None:
    sampler_list = get_sampler_list()
    scheduler_list = get_noise_schedulers()

    with gr.Blocks(title="Colorize Diffusion", css="div.gradio-container{ max-width: unset !important; }") as block:
        with gr.Row():
            gr.Markdown(f"## Colorize Diffusion")

        with gr.Row():
            with gr.Column():
                with gr.Row(visible=opt.enable_text_manipulation):
                    target = gr.Textbox(label="Target prompt", value="", scale=2)
                    anchor = gr.Textbox(label="Anchor prompt", value="", scale=2)
                    control = gr.Textbox(label="Control prompt", value="", scale=2)
                with gr.Row(visible=opt.enable_text_manipulation):
                    target_scale = gr.Slider(label="Target scale", value=0.0, minimum=0, maximum=15.0, step=0.25, scale=2)
                    ts0 = gr.Slider(label="Threshold 0", value=0.5, minimum=0, maximum=1.0, step=0.01)
                    ts1 = gr.Slider(label="Threshold 1", value=0.55, minimum=0, maximum=1.0, step=0.01)
                    ts2 = gr.Slider(label="Threshold 2", value=0.65, minimum=0, maximum=1.0, step=0.01)
                    ts3 = gr.Slider(label="Threshold 3", value=0.95, minimum=0, maximum=1.0, step=0.01)
                with gr.Row(visible=opt.enable_text_manipulation):
                    enhance = gr.Checkbox(label="Enhance manipulation", value=False)
                    add_prompt = gr.Button(value="Add")
                    clear_prompt = gr.Button(value="Clear")
                    vis_button = gr.Button(value="Visualize")
                text_prompt = gr.Textbox(label="Final prompt", value="", lines=3, visible=opt.enable_text_manipulation)

                with gr.Row():
                    sketch_img = gr.Image(label="Sketch", type="pil", height=256)
                    reference_img = gr.Image(label="Reference", type="pil", height=256)
                    background_img = gr.Image(label="Background (optional)", type="pil", height=256)

                with gr.Row():
                    style_enhance = gr.Checkbox(label="Style enhance", value=False)
                    bg_enhance = gr.Checkbox(label="BG enhance (mask guided)", value=False)
                    fg_enhance = gr.Checkbox(label="FG enhance (mask guided)", value=False)
                    injection = gr.Checkbox(label="Attention injection", value=False)
                with gr.Row():
                    gs_r = gr.Slider(label="Reference guidance scale", minimum=1, maximum=15.0, value=4.0, step=0.5)
                    strength = gr.Slider(label="Reference strength", minimum=0, maximum=1, value=1, step=0.05)
                    fg_strength = gr.Slider(label="Foreground strength", minimum=0, maximum=1, value=1, step=0.05)
                    bg_strength = gr.Slider(label="Background strength", minimum=0, maximum=1, value=1, step=0.05)
                with gr.Row():
                    gs_s = gr.Slider(label="Sketch guidance scale", minimum=1, maximum=5.0, value=1.0, step=0.1)
                    ctl_scale = gr.Slider(label="Sketch strength", minimum=0, maximum=3, value=1, step=0.1)
                    mask_scale = gr.Slider(label="Background factor", minimum=0, maximum=2, value=1, step=0.05)
                    merge_scale = gr.Slider(label="Merging scale", minimum=0, maximum=1, value=0, step=0.05)
                with gr.Row():
                    low_scale = gr.Slider(label="Semantics crossattn scale",
                                          minimum=0, maximum=1, step=0.05, value=1)
                    middle_scale = gr.Slider(label="Color crossattn scale", minimum=0,
                                             maximum=1, step=0.05, value=1)
                    enc_scale = gr.Slider(label="Encoder crossattn scale",
                                           minimum=0, maximum=1, step=0.05, value=1)
                    fg_disentangle_scale = gr.Slider(label="Disentangle scale",
                                                     minimum=0, maximum=1, step=0.05, value=1)

                with gr.Accordion("Advanced Setting", open=True):
                    with gr.Row():
                        crop = gr.Checkbox(label="Crop result", value=False, scale=1)
                        remove_fg = gr.Checkbox(label="Remove foreground in background input", value=False, scale=2)
                        rmbg = gr.Checkbox(label="Remove background in result", value=False, scale=2)
                    with gr.Row():
                        injection_control_scale = gr.Slider(label="Injection fidelity (sketch)", minimum=0.0, maximum=2.0,
                                                            value=0, step=0.05)
                        injection_fidelity = gr.Slider(label="Injection fidelity (reference) ", minimum=0.0, maximum=1.0,
                                                       value=0.5, step=0.05)
                        injection_start_step = gr.Slider(label="Injection start step ", minimum=0.0, maximum=1.0,
                                                       value=0, step=0.05)
                    with gr.Row():
                        start_step = gr.Slider(label="Reference guidance start step", minimum=0, maximum=1, value=0,
                                               step=0.05)
                        end_step = gr.Slider(label="Reference guidance end step", minimum=0, maximum=1, value=1,
                                             step=0.05)
                        no_start_step = gr.Slider(label="No guidance start step", minimum=-0.05, maximum=1, value=-0.05,
                                                  step=0.05)
                        no_end_step = gr.Slider(label="No guidance end step", minimum=-0.05, maximum=1, value=-0.05,
                                                step=0.05)

                with gr.Accordion("Accurate control on crossattn scale (only for SD2.1)", open=True):
                    accurate = gr.Checkbox(label="Activate accurate crossattn control", value=False)
                    with gr.Row():
                        attn0 = gr.Slider(label="0.down0.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn1 = gr.Slider(label="1.down0.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn2 = gr.Slider(label="2.down1.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn3 = gr.Slider(label="3.down1.attn1", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn4 = gr.Slider(label="4.down2.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn5 = gr.Slider(label="5.down2.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn6 = gr.Slider(label="6.middle", minimum=0, maximum=1, step=0.05, value=1)
                        attn7 = gr.Slider(label="7.up2.attn0", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn8 = gr.Slider(label="8.up2.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn9 = gr.Slider(label="9.up2.attn2", minimum=0, maximum=1, step=0.05, value=1)
                        attn10 = gr.Slider(label="10.up1.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn11 = gr.Slider(label="11.up1.attn1", minimum=0, maximum=1, step=0.05, value=1)
                    with gr.Row():
                        attn12 = gr.Slider(label="12.up1.attn2", minimum=0, maximum=1, step=0.05, value=1)
                        attn13 = gr.Slider(label="13.up0.attn0", minimum=0, maximum=1, step=0.05, value=1)
                        attn14 = gr.Slider(label="14.up0.attn1", minimum=0, maximum=1, step=0.05, value=1)
                        attn15 = gr.Slider(label="15.up0.attn2", minimum=0, maximum=1, step=0.05, value=1)

            with gr.Column():
                result_gallery = gr.Gallery(
                    label='Output', show_label=False, elem_id="gallery", preview=True, format="png"
                )
                run_button = gr.Button(value="Run")

                with gr.Row():
                    mask_ts = gr.Slider(label="Mask threshold", minimum=0., maximum=1., value=0.5, step=0.01)
                    mask_ss = gr.Slider(label="Sketch mask threshold", minimum=0., maximum=1., value=0.05, step=0.01)
                    pad_scale = gr.Slider(label="Padding scale", minimum=1, maximum=2, value=1, step=0.05)

                with gr.Row():
                    sd_model = gr.Dropdown(choices=get_checkpoints(), label="Models", value=get_checkpoints()[0])
                    extractor_model = gr.Dropdown(choices=line_extractor_list,
                                              label="Line extractor", value=default_line_extractor)
                    mask_model = gr.Dropdown(choices=mask_extractor_list, label="Reference mask extractor",
                                             value=default_mask_extractor)
                with gr.Row():
                    sampler = gr.Dropdown(choices=sampler_list, value="diffuser_dpm", label="Sampler")
                    scheduler = gr.Dropdown(choices=scheduler_list, value=scheduler_list[0], label="Noise scheduler")
                    preprocessor = gr.Dropdown(choices=["none", "extract", "invert", "invert-webui"],
                                               label="Sketch preprocessor", value="invert")

                with gr.Row():
                    bs = gr.Slider(label="Batch size", minimum=1, maximum=8, value=1, step=1, scale=1)
                    width = gr.Slider(label="Width", minimum=512, maximum=1536, value=512, step=64, scale=2)
                with gr.Row():
                    step = gr.Slider(label="Step", minimum=1, maximum=100, value=20, step=1, scale=1)
                    height = gr.Slider(label="Height", minimum=512, maximum=1536, value=512, step=64, scale=2)
                # with gr.Row():
                #     # Only to check 512x512 images.
                #     vattn = gr.Checkbox(label="Visualize crossattn", value=False)
                #     vh = gr.Slider(label="Visualize region (#H)", minimum=0, maximum=15, step=1, value=0)
                #     vw = gr.Slider(label="Visualize region (#W)", minimum=0, maximum=15, step=1, value=0)

                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=MAXM_INT32, step=1, value=-1)

                with gr.Row():
                    reuse_seed = gr.Button(value="Reuse seed")
                    random_seed = gr.Button(value="Random seed")
                    update_ckpts = gr.Button(value="Update checkpoints")
                    # split = gr.Button(value="Highlight selected regions")

                with gr.Row():
                    vae_fp16 = gr.Button(value="fp16 vae")
                    vae_fp32 = gr.Button(value="fp32 vae")
                    fp16 = gr.Button(value="fp16 unet")
                    fp32 = gr.Button(value="fp32 unet")

                with gr.Row():
                    deterministic = gr.Checkbox(label="Deterministic batch seed", value=False)
                    save_memory = gr.Checkbox(label="Save memory", value=False)
                    return_inter = gr.Checkbox(label="Check intermediates (only for ddim)", value=False)

        add_prompt.click(fn=apppend_prompt,
                         inputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt],
                         outputs=[target, anchor, control, target_scale, enhance, ts0, ts1, ts2, ts3, text_prompt])
        clear_prompt.click(fn=clear_prompts,
                           outputs=[text_prompt])
        sd_model.input(fn=load_model, inputs=[sd_model])
        extractor_model.input(fn=switch_extractor, inputs=[extractor_model])
        mask_model.input(fn=switch_mask_extractor, inputs=[mask_model])
        reuse_seed.click(fn=get_last_seed, outputs=[seed])
        random_seed.click(fn=reset_random_seed, outputs=[seed])
        update_ckpts.click(fn=update_models, outputs=[sd_model])

        fp16.click(fn=switch_to_fp16)
        fp32.click(fn=switch_to_fp32)
        vae_fp16.click(fn=switch_vae_to_fp16)
        vae_fp32.click(fn=switch_vae_to_fp32)

        ips = [style_enhance, bg_enhance, fg_enhance, fg_disentangle_scale,
               bs, sketch_img, reference_img, background_img, mask_ts, mask_ss, gs_r, gs_s, ctl_scale,
               fg_strength, bg_strength, merge_scale, mask_scale, height, width, seed, save_memory, step, injection,
               remove_fg, rmbg, injection_control_scale, injection_fidelity, injection_start_step, crop, pad_scale,
               start_step, end_step, no_start_step, no_end_step, return_inter, sampler, scheduler, preprocessor,
               deterministic, text_prompt, target, anchor, control, target_scale, ts0, ts1, ts2, ts3,
               enhance, accurate, enc_scale, middle_scale, low_scale, strength, attn0, attn1, attn2, attn3,
               attn4, attn5, attn6, attn7, attn8, attn9, attn10, attn11, attn12, attn13, attn14, attn15]

        run_button.click(fn=inference, inputs=ips, outputs=[result_gallery])
        # split.click(fn=split_sketch, inputs=[sketch_img, vh, vw], outputs=[result_gallery])
        vis_button.click(fn=visualize,
                         inputs=[reference_img, text_prompt, control, ts0, ts1, ts2, ts3],
                         outputs=[result_gallery])
        block.load(fn=update_models, outputs=[sd_model])

    block.launch(
        server_name = opt.server_name,
        share = opt.share,
        server_port = opt.server_port,
        show_error = not opt.not_show_error
    )


if __name__ == '__main__':
    opt = app_optiosn()
    load_model(get_checkpoints()[0])
    switch_extractor(default_line_extractor)
    switch_mask_extractor(default_mask_extractor)
    interface = init_inerface(opt)