import os
import time

import torch
import torchvision
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from ldm.util import default
from accelerate import Accelerator

MAXM_SAMPLE_SIZE = 16
ckpt_fmt = "safetensors"


def format_time(second):
    s = int(second)
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


class CustomCheckpoint:
    def __init__(
            self,
            save_first_step,
            save_freq_step,
            not_save_weight_only,
            ckpt_path,
            start_save_ep,
            save_freq,
            top_k,
            **kwargs
    ):
        self.save_first_step = save_first_step
        self.save_freq_step = save_freq_step
        self.save_weight_only = not not_save_weight_only
        self.ckpt_path = ckpt_path
        self.start_save_ep = start_save_ep
        self.save_freq = save_freq
        self.top_k = top_k
        self.prev_ckpts = []
        self.prev_time = time.time()


    def on_train_start(self, trainer: Accelerator):
        if self.save_first_step:
            filename = os.path.join(self.ckpt_path, f'latest')
            trainer.wait_for_everyone()
            trainer.save_state(filename)

            if trainer.is_local_main_process:
                message = f"Saving latest model to {filename}"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


    def on_train_batch_end(self, trainer: Accelerator, global_step, batch_idx):
        if global_step > 2 and batch_idx % self.save_freq_step == 0:
            filename = os.path.join(self.ckpt_path, f'latest')
            trainer.wait_for_everyone()
            trainer.save_state(filename)

            if trainer.is_local_main_process:
                curtime = time.time()
                interval = curtime - self.prev_time
                self.prev_time = curtime
                message = (f"***** Saving latest model to {os.path.abspath(filename)}, "
                           f"saving interval: {format_time(interval)} *****")
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


    def on_train_epoch_end(self, trainer: Accelerator, current_epoch):
        if current_epoch >= self.start_save_ep and current_epoch % self.save_freq == 0:
            filename = os.path.abspath(os.path.join(self.ckpt_path, f"epoch-{current_epoch}"))
            trainer.wait_for_everyone()
            trainer.save_state(filename)

            if trainer.is_local_main_process:
                if len(self.prev_ckpts) >= self.top_k:
                    import shutil
                    filename = self.prev_ckpts.pop(0)
                    if os.path.exists(filename):
                        shutil.rmtree(filename)
                self.prev_ckpts.append(filename)
                message = f"***** Saving latest model to {filename} *****"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


class ConsoleLogger:
    def __init__(
            self,
            print_freq,
            ckpt_path,
            batch_per_epoch,
            stage="train",
            **kwargs
    ):
        self.step_frequcy = print_freq
        self.ckpt_path = ckpt_path
        self.batch_per_epoch = batch_per_epoch

        current_time = time.time()
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time
        self.total_time = 0

        self.stage = stage

    def on_train_batch_end(
            self,
            batch_idx,
            loss_dict,
            current_epoch,
            max_epoch,
            global_step,
            learning_rate,
            **kwargs
    ):
        if batch_idx % self.step_frequcy == 0:
            current_time = time.time()
            fmt_curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            iter_time = current_time - self.pre_iter_time
            self.total_time += iter_time
            self.pre_iter_time = current_time

            message = f"{fmt_curtime}"
            message += f", iter_time: {format_time(iter_time)}"
            message += f", total_time: {format_time(self.total_time)}"
            message += f", epoch: [{current_epoch}/{max_epoch}]"
            message += f", step: [{batch_idx}/{self.batch_per_epoch}]"
            message += f", global_step: {global_step}"
            message += f", lr: {learning_rate:.7f}"

            for label in loss_dict:
                message += f', {label}: {loss_dict[label]:.6f}'

            tqdm.write(message)
            train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()

    def on_train_epoch_end(self, current_epoch, global_step, learning_rate):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        epoch_time = current_time - self.pre_epoch_time
        self.total_time += iter_time
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time

        message = "{ "
        message += f"Epcoh {current_epoch} finished"
        message += f",\tglobal_step: {global_step}"
        message += f",\ttotal_time: {format_time(self.total_time)}"
        message += f",\tepoch_time: {format_time(epoch_time)}"
        message += f", current_lr: {learning_rate:.7f}"
        message += " }"

        tqdm.write(message)
        train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
        train_log.write(message + '\n\n')
        train_log.close()


class ImageLogger:
    def __init__(
            self,
            ckpt_path,
            save_freq_step=1,
            sample_num=None,
            clamp=True,
            increase_log_steps=True,
            batch_size=None,
            rescale=True,
            disabled=False,
            check_memory_use=False,
            log_on_batch_idx=True,
            log_first_step=True,
            sampler="dpm",
            step=20,
            guidance_scale=1.0,
            save_input=False,
            load_checkpoint=None,
            pretrained=None,
            log_img_step=None,
            **kwargs
    ):
        self.clamp = clamp
        self.rescale = rescale
        self.save_input = save_input
        self.log_on_batch_idx = log_on_batch_idx
        self.disabled = disabled or sample_num == 0
        self.sample_num = default(sample_num, batch_size)
        self.batch_freq = default(log_img_step, save_freq_step)
        self.save_path = os.path.join(ckpt_path, kwargs.pop("mode"))
        self.log_first_step = log_first_step if not check_memory_use else False

        self.sampling_params = {
            "sample": guidance_scale == 1.,
            "unconditional_guidance_scale": guidance_scale,
            "sampler": sampler,
            "step": step,
        }

        if load_checkpoint is None and pretrained is None and not check_memory_use:
            self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        else:
            self.log_steps = []
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]


    def log_local(self, images, global_step, current_epoch, batch_idx, is_train):
        def save_image(img, path):
            if self.rescale:
                img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            img = img.permute(1, 2, 0).squeeze(-1)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(path)

        for k in images:
            dirpath = os.path.join(self.save_path, k)
            os.makedirs(dirpath, exist_ok=True)

            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu().float()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
            if len(images[k].shape) == 3:
                images[k] = images[k].unsqueeze(0)

            if is_train:
                # save grid images during training
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}.png"
                path = os.path.join(dirpath, filename)
                save_image(grid, path)

            else:
                # save images separately during testing
                for idx in range(len(images[k])):
                    img = images[k][idx]
                    filename = f"{batch_idx}_{idx}.png"
                    path = os.path.join(dirpath, filename)
                    save_image(img, path)

    def log_img(self, model, batch, batch_idx, global_step=None, current_epoch=None):
        is_train = model.training
        if (hasattr(model, "log_images") and callable(model.log_images) and self.sample_num > 0) or not is_train:
            if is_train:
                model.eval()

            images = model.log_images(
                N = min(self.sample_num, MAXM_SAMPLE_SIZE) if is_train else self.sample_num,
                batch = batch,
                return_inputs = is_train or self.save_input,
                **self.sampling_params,
            )

            self.log_local(images, global_step, current_epoch, batch_idx, is_train)
            if is_train:
                model.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, model, global_step, current_epoch, batch, batch_idx, **kwargs):
        check_idx = batch_idx if self.log_on_batch_idx else global_step
        if not self.disabled and global_step > 0 and self.check_frequency(check_idx):
            self.log_img(model, batch, batch_idx, global_step, current_epoch)

    def on_test_batch_end(self, model, batch, batch_idx):
        self.log_img(model, batch, batch_idx)