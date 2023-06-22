import gradio as gr
import torch
import numpy as np
import cv2
import yaml
from PIL import Image
import utils

config_file = "configs/inference/token.yaml"
with open(config_file, 'r') as f:
    configs = yaml.safe_load(f.read())
print(f"Loaded model config from {config_file}")
model = utils.instantiate_from_config(configs['model']).eval()
#model.cond_stage_model = model.cond_stage_model.cuda()
model = model.cuda()
origin_res = (512, 512)
mani_params_num = 4
mani_params = [{} for _ in range(mani_params_num)]
maxium_resolution = 2048

def inference(sketch, reference, scale, resolution, seed, sampler, steps):
    global origin_res
    global model
    origin_res = sketch.size
    controls, targets, anchors, thresholdlists, target_scales, enhances = parse_manipulation_params()
    result = model.generate_image(
        sketch = sketch.convert('L'),
        reference = reference,
        unconditional_guidance_scale = scale,
        resolution = resolution,
        enhance = enhances,
        controls = controls,
        targets = targets,
        anchors = anchors,
        thresholds_list = thresholdlists,
        target_scales = target_scales,
        seed = seed,
        sampler = sampler,
        steps = steps
    )
#    result = cv2.resize(result, origin_res, interpolation=cv2.INTER_LINEAR)
    return [result]


def set_list_value(index, key, value):
    global mani_params
    mani_params[index][key].value = value


def parse_manipulation_params():
    global mani_params
    controls = []
    targets = []
    anchors = []
    target_scales = []
    thresholdlists = []
    enhances = []
    for param in mani_params:
        if param['scale'].value == 0.0: continue
        controls.append(param['ctrl_prompt'].value)
        targets.append(param['target_prompt'].value)
        anchors.append(param['anchor_prompt'].value)
        target_scales.append(param['scale'].value)
        thresholdlists.append([param['ts_0'].value, param['ts_1'].value, param['ts_2'].value, param['ts_3'].value])
        enhances.append(param['enhance'].value)

    return controls, targets, anchors, thresholdlists, target_scales, enhances


def get_heatmaps(reference, height, width):
    global model
    # the image here is for reference
    controls, targets, anchors, thresholdlists, target_scales, enhances = parse_manipulation_params()
    v = model.get_tokens(reference)
    cls_token = v[:,0].unsqueeze(0)
    all_heatmaps = []
    global_scales = []
    print("controls:", controls)
    print("targets:", targets)
    print("anchors:", anchors)
    print("target scales:", target_scales)
    print("thresholds list:", thresholdlists)
    print("enhance list:", enhances)
    for control, target, anchor, target_scale, thresholds, enhance in zip(controls, targets, anchors, target_scales, thresholdlists,
                                                                  enhances):
        local_v = v[:, 1:]
        c, t = model.cond_stage_model.encode_text([control]), model.cond_stage_model.encode_text([target])
        global_scale = cls_token @ t.mT
        global_scales.append(global_scale)
        scale = model.get_projections(local_v, c)
        scale = scale.permute(0, 2, 1).view(1, 1, 16, 16)
        scale = torch.nn.functional.interpolate(scale, size=(height, width), mode="bicubic").squeeze(0).view(1,
                                                                                                             height * width)
        # calculate heatmaps
        heatmaps = []
        for threshold in thresholds:
            heatmap = model.get_heatmap(scale, threshold=threshold)
            heatmap = heatmap.view(1, height, width).permute(1, 2, 0).cpu().numpy()
            heatmap = (heatmap * 255.).astype(np.uint8)
            heatmaps.append(heatmap)
        all_heatmaps.append(heatmaps)
        # update image tokens
        v = model.manipulate_step(
            v=v, 
            t=target, 
            target_scale=target_scale,
            a=anchor,
            c=control, 
            enhance=enhance, 
            thresholds=thresholds)
    return all_heatmaps, target_scales, global_scales

line_color = (0, 0, 0)
length = 16

def visualize_heatmaps(reference):
    if reference is None:
        return []
    size = reference.size
    if size[0] > maxium_resolution or size[1] > maxium_resolution:
        if size[0] > size[1]:
            size = (2048, int(2048.0 / size[0] * size[1]))
        else:
            size = (int(2048.0 / size[1] * size[0]), 2048)
        reference.resize(size, Image.BICUBIC)
    scale_maps_list, target_scales, global_scales = get_heatmaps(reference, size[1], size[0])
    results = []
    reference = np.array(reference)
    for scale_maps, t_s, g_s in zip(scale_maps_list, target_scales, global_scales):
        scale_map = scale_maps[0] + scale_maps[1] + scale_maps[2] + scale_maps[3]
        heatmap = cv2.cvtColor(cv2.applyColorMap(scale_map, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        result = cv2.addWeighted(reference, 0.3, heatmap, 0.7, 0)
        hu = size[1] // length
        wu = size[0] // length
        for i in range(16):
            result[i * hu, :] = line_color
        for i in range(16):
            result[:, i * wu] = line_color
        title = "Target Scale: " + str(t_s) + ", Global Scale: " + str(g_s)
        results.append((result, title))
    return results


def init_inerface() -> None:
    global mani_params
    block = gr.Blocks().queue(concurrency_count=3)
    with block:
        with gr.Row():
            gr.Markdown("## Colorize Diffusion")
        with gr.Row():
            with gr.Column():
                sketch_img = gr.Image(label="Sketch", source='upload', type="pil")
                reference_img = gr.Image(label="Reference", source='upload', type="pil")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced Setting", open=False):
                    sampler = gr.Radio(choices=["ddpm", "ddim", "dpm"], type="value", value="dpm", label="Sampler")
                    steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=20, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=5.0, step=0.1)
                    resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=2048, value=512, step=64)
                    #eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    #ema = gr.Checkbox(label="Use EMA Scope", value=False)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            with gr.Column():
                ref_gallery = gr.Gallery(label='Position Weights', show_label=False, elem_id="gallery").style(grid=1,
                                                                                                              height='auto')
                with gr.Group():      
                    for index, param in enumerate(mani_params):  
                        with gr.Accordion("Param Group " + str(index), open=False):
                            with gr.Group():
                                with gr.Row():
                                    param['target_prompt'] = gr.Textbox(label="Target Prompt")
                                    param['target_prompt'].change(fn=lambda x, i=index: set_list_value(i, 'target_prompt', x),
                                                inputs=param['target_prompt'])
                                    param['anchor_prompt'] = gr.Textbox(label="Anchor Prompt")
                                    param['anchor_prompt'].change(fn=lambda x, i=index: set_list_value(i, 'anchor_prompt', x),
                                                inputs=param['anchor_prompt'])
                                    param['ctrl_prompt'] = gr.Textbox(label="Control Prompt")
                                    param['ctrl_prompt'].change(fn=lambda x, i=index: set_list_value(i, 'ctrl_prompt', x),
                                                inputs=param['ctrl_prompt'])
                                    param['enhance'] = gr.Checkbox(label="Enhance", value=False)
                                    param['enhance'].change(fn=lambda x, i=index: set_list_value(i, 'enhance', x),
                                                inputs=param['enhance'])
                                    param['ts_0'] = gr.Number(label="Threshold 0", value=0.5)
                                    param['ts_0'].change(fn=lambda x, i=index: set_list_value(i, 'ts_0', x), inputs=param['ts_0'])
                                    param['ts_1'] = gr.Number(label="Threshold 1", value=0.55)
                                    param['ts_1'].change(fn=lambda x, i=index: set_list_value(i, 'ts_1', x), inputs=param['ts_1'])
                                    param['ts_2'] = gr.Number(label="Threshold 2", value=0.65)
                                    param['ts_2'].change(fn=lambda x, i=index: set_list_value(i, 'ts_2', x), inputs=param['ts_2'])
                                    param['ts_3'] = gr.Number(label="Threshold 3", value=0.95)
                                    param['ts_3'].change(fn=lambda x, i=index: set_list_value(i, 'ts_3', x), inputs=param['ts_3'])
                                    param['scale'] = gr.Slider(label="Manipulation Scale", minimum=0, maximum=15.0, value=0.0,
                                                step=0.1)
                                    param['scale'].change(fn=lambda x, i=index: set_list_value(i, 'scale', x),
                                                inputs=param['scale'])
                vis_button = gr.Button(value="Visualize")
                vis_button.click(fn=visualize_heatmaps, inputs=[reference_img], outputs=[ref_gallery])
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=1,
                                                                                                       height='auto')
        ips = [sketch_img, reference_img, scale, resolution, seed, sampler, steps]
        run_button.click(fn=inference, inputs=ips, outputs=[result_gallery])
    block.launch(server_name='localhost')


# parser = argparse.ArgumentParser()
# parser.add_argument('-num', type=int, help='An integer number', default=4)
# args = parser.parse_args()
interface = init_inerface()
