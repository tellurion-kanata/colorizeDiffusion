import gradio as gr

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Colorize Diffusion")
    with gr.Row():
        with gr.Column():
            sketch_img = gr.Image(label="Sketch", source='upload', type="numpy")
            reference_img = gr.Image(label="Reference", source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=2048, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

    ips = [sketch_img, reference_img, prompt, image_resolution, detect_resolution, ddim_steps, strength, scale, seed, eta]
    #run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='localhost')