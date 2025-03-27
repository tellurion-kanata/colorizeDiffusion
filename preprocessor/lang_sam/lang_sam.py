import numpy as np
import torch
import torchvision.transforms.functional as tf

from PIL import Image
from .models.gdino import GDINO
from .models.sam import SAM
from backend.functool import pad_image, resize

class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None):
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path)
        self.gdino = GDINO()
        self.gdino.build_model()

    def cpu(self):
        self.sam.model = self.sam.model.cpu()
        self.gdino.model = self.gdino.model.cpu()

    def cuda(self):
        self.sam.model = self.sam.model.cuda()
        self.gdino.model = self.gdino.model.cuda()

    def proceed(self, pil_x: Image.Image, th=None, tw=None, threshold = 0.3, *args, **kwargs):
        w, h = pil_x.size
        text = ["character"]
        mask = torch.Tensor(
            self.predict([pil_x], text, box_threshold=threshold)[0]["masks"]
        ).unsqueeze(0)

        if th and tw:
            r = min(th / h, tw / w)
            mask, _ = pad_image(resize((int(h * r), int(w * r)))(mask), th, tw)
        mask = tf.gaussian_blur(mask.cuda(), kernel_size=[7, 7], sigma=[1, 1])
        return mask

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                processed_result["boxes"] = result["boxes"].cpu().numpy()
                processed_result["scores"] = result["scores"].cpu().numpy()
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            # print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            # print(f"Predicted {len(all_results)} masks")
        return all_results


if __name__ == "__main__":
    model = LangSAM()
    out = model.predict(
        [Image.open("./assets/food.jpg"), Image.open("./assets/car.jpeg")],
        ["food", "car"],
    )
    print(out)
