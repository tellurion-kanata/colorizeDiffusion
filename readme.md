# ColorizeDiffusion: Adjustable Sketch Colorization with Reference Image and Text

![img](assets/teaser.png)

(May. 2024) 
Paper link for this repository: [ColorizeDiffusion](https://arxiv.org/abs/2401.01456), we will update a new paper for the updated model.

New model weights link: https://huggingface.co/tellurion/colorizer (release soon).

## Implementation Details
The repository offers the updated implementation of ColorizeDiffusion.  
Now there is only the noisy model introduced in the paper, which utilizes the local tokens.

## Getting Start
To utilize the code in this repository, ensure that you have installed the required dependencies as specified in the requirements.

### To install and run:
```shell
conda env create -f environment.yaml
conda activate hf
```

## User Interface:
We also provided a Web UI based on Gradio UI. To run it, just:
```shell
python -u gradio_ui.py
```
Then you can browse the UI in http://localhost:7860/.

### Inference:

#### Inference options:
| Options                   | Description                                                                       |
|:--------------------------|:----------------------------------------------------------------------------------|
| Crossattn scale           | Used to diminish all kinds of artifacts caused by the distribution problem.       |
| Pad reference             | Activate to use "pad reference with margin"                                       |
| Pad reference with margin | Used to diminish spatial entanglement, pad reference to T times of current width. |
| Reference guidance scale  | Classifier-free guidance scale of the reference image, suggested 5.               |
| Sketch guidance scale     | Classifier-free guidance scale of the sketch image, suggested 1.                  |
| Attention injection       | Strengthen similarity with reference.                                             |
| Visualize                 | Used for local manipulation. Visualize the regions selected by each threshold.    |

For artifacts like spatial entanglement:
1. Activate **Pad reference** and increase **Pad reference with margin** to around 1.5, or
2. Reduce **Overall crossattn scale** to 0.4-0.8. (Best for handling all kinds of artifacts caused by the distribution problem, but accordingly degrade the similarity with referneces)

We offer a precise control of crossattn scales, check **Accurate control** part. 

Q: Why padding margin is useful?  
A: Margins are embedded to "pure white backgrounds" and suppress the generation of backgrounds. We notice that models trained on large-scale datasets sometimes hard to generate white backgroudns, especially using image prompt and anime-style data.

When using stylized image like ***The Starry Night*** for style transfer, try **Attention injection** with **Karras**-based sampler.
![img](assets/style%20transfer.png)

### Manipulation:
The colorization results can be manipulated using text prompts.

For local manipulations, a visualization is provided to show the correlation between each prompt and tokens in the reference image.


The manipulation result and correlation visualization of the settings:
    
    Target prompt: the girl's blonde hair
    Anchor prompt the girl's brown hair
    Control prompt the girl's brown hair, 
    Target scale: 8
    Enhanced: false
    Thresholds: 0.5、0.55、0.65、0.95

![img](assets/preview1.png)
![img](assets/preview2.png)
As you can see, the manipluation unavoidably changed some unrelated regions as it is taken on the reference embeddings.

#### Manipulation options:
| Options                   | Description                                                                                                                                                                                                       |
| :-----                    |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Group index               | The index of selected manipulation sequences's parameter group.                                                                                                                                                   |
| Target prompt             | The prompt used to specify the desired visual attribute for the image after manipulation.                                                                                                                         |
| Anchor prompt             | The prompt to specify the anchored visaul attribute for the image before manipulation.                                                                                                                            |
| Control prompt            | Used for local manipulation (crossattn-based models). The prompt to specify the target regions.                                                                                                                   |
| Enhance                   | Specify whether this manipulation should be enhanced or not. (More likely to influence unrelated attribute).                                                                                                      |
| Target scale              | The scale used to progressively control the manipulation.                                                                                                                                                         |
| Thresholds                | Used for local manipulation (crossattn-based models). Four hyperparameters used to reduce the influnece on irrelevant visual attributes, where 0.0 < threshold 0 < threshold 1 < threshold 2 < threshold 3 < 1.0. |
| \<Threshold0 				| Select regions most related to control prompt. Indicated by deep blue.                                                                                                                                            |
| Threshold0-Threshold1     | Select regions related to control prompt. Indicated by blue.                                                                                                                                                      |
| Threshold1-Threshold2		| Select neighbouring but unrelated regions. Indicated by green.                                                                                                                                                    |
| Threshold2-Threshold3		| Select unrelated regions. Indicated by orange.                                                                                                                                                                    |
| \>Threshold3				| Select most unrelated regions. Indicated by brown.                                                                                                                                                                |
|Add| Click add to save current manipulation in the sequence.                                                                                                                                                           |

## License
This project is licensed under the [cc-nc-sa 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) and [stable diffusion](https://huggingface.co/spaces/CompVis/stable-diffusion-license).