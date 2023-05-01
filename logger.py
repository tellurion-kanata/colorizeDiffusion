import os
import time
import utils
import torch
import torchvision
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

maxm_sample_size = 16
OPENAI_MEAN = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
OPENAI_STD = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)

def default(opt, v, d=None):
    if hasattr(opt, v) and getattr(opt, v) is not None:
        return getattr(opt, v)
    return d


class CustomCheckpoint(Callback):
    def __init__(self, opt):
        self.save_first_step = opt.save_first_step
        self.save_freq = opt.save_freq_step
        self.save_weight_only = not opt.not_save_weight_only
        self.ckpt_path = opt.ckpt_path

        self.start_save_ep = opt.start_save_ep
        self.top_k = opt.top_k
        self.prev_ckpts = []

        self.prev_time = time.time()

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if self.save_first_step:
            filename = os.path.join(self.ckpt_path, 'latest.ckpt')
            message = f"Saving latest model to {filename}"
            trainer.save_checkpoint(filename, weights_only=True)

            tqdm.write(message)
            train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 2 and batch_idx % self.save_freq == 0:
            curtime = time.time()
            interval = curtime - self.prev_time
            self.prev_time = curtime

            filename = os.path.join(self.ckpt_path, 'latest.ckpt')
            message = f"Saving latest model to {filename}, saving interval: {utils.format_time(interval)}"

            tqdm.write(message)
            trainer.save_checkpoint(filename, weights_only=self.save_weight_only)

            train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_save_ep:
            if len(self.prev_ckpts) == self.top_k:
                filename = self.prev_ckpts.pop(0)
                os.remove(filename)
                weights_path = filename.replace(".ckpt", "_model.ckpt")
                if os.path.exists(weights_path):
                    os.remove(weights_path)

            filename = os.path.join(self.ckpt_path, f"epoch-{trainer.current_epoch}.ckpt")
            message = f"Saving latest model to {filename}"

            tqdm.write(message)
            trainer.save_checkpoint(filename, weights_only=self.save_weight_only)

            self.prev_ckpts.append(filename)
            train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()


class ConsoleLogger(Callback):
    def __init__(self, step_frequency, ckpt_path, batch_per_epoch, stage="train"):
        super().__init__()
        self.step_frequcy = step_frequency
        self.ckpt_path = ckpt_path
        self.batch_per_epoch = batch_per_epoch

        current_time = time.time()
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time
        self.total_time = 0

        self.stage = stage

    def refine(self, lossdict):
        delete_keys = []
        no_stage_keys = []
        return_dict = {}

        for label in lossdict:
            if label.find("step") == -1 or label.find(self.stage) == -1:
                delete_keys.append(label)
            else:
                no_stage_keys.append(label.replace(self.stage+"/", "").replace("_step", ""))
        for key in delete_keys:
            lossdict.pop(key)
        for key in no_stage_keys:
            return_dict[key] = lossdict[self.stage+"/"+key+"_step"]
        return return_dict

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.step_frequcy == 0:
            current_time = time.time()
            fmt_curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            iter_time = current_time - self.pre_iter_time
            self.total_time += iter_time
            self.pre_iter_time = current_time

            message = f"{fmt_curtime}"
            message += f", iter_time: {utils.format_time(iter_time)}"
            message += f", total_time: {utils.format_time(self.total_time)}"
            message += f", epoch: [{trainer.current_epoch}/{trainer.max_epochs}]"
            message += f", step: [{batch_idx}/{self.batch_per_epoch}]"
            message += f", global_step: {trainer.global_step}"
            if len(trainer.optimizers) > 1:
                message += f", g_lr: {trainer.optimizers[0].param_groups[0]['lr']:.7f}"
                message += f", d_lr: {trainer.optimizers[1].param_groups[0]['lr']:.7f}"
            else:
                message += f", lr: {trainer.optimizers[0].param_groups[0]['lr']:.7f}"

            loss_dict = self.refine(trainer.callback_metrics)
            for label in loss_dict:
                message += f', {label}: {loss_dict[label]:.6f}'

            tqdm.write(message)
            train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        epoch_time = current_time - self.pre_epoch_time
        self.total_time += iter_time
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time

        message = "{ "
        message += f"Epcoh {trainer.current_epoch} finished"
        message += f",\tglobal_step: {trainer.global_step}"
        message += f",\ttotal_time: {utils.format_time(self.total_time)}"
        message += f",\tepoch_time: {utils.format_time(epoch_time)}"
        if len(trainer.optimizers) > 1:
            message += f",\tcurrent_g_lr: {trainer.optimizers[0].param_groups[0]['lr']:<.7f}"
            message += f",\tcurrent_d_lr: {trainer.optimizers[1].param_groups[0]['lr']:<.7f}"
        else:
            message += f", current_lr: {trainer.optimizers[0].param_groups[0]['lr']:.7f}"
        message += " }"

        tqdm.write(message)
        train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
        train_log.write(message + '\n\n')
        train_log.close()


class ImageLogger(Callback):
    def __init__(self, batch_frequency, save_path, sample_num, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, check_memory_use=False, log_on_batch_idx=True, log_first_step=True,
                  ddim_sample=False, ddim_sample_step=200, guidance_scale=1.0, guidance_label="reference",
                 save_input=False, use_ema=False, resume=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.sample_num = sample_num
        if not resume and not check_memory_use:
            self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        else:
            self.log_steps = []
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step if not check_memory_use else False
        self.save_path = save_path
        self.guidance_scale = guidance_scale
        self.guidance_label = guidance_label
        self.use_ema = use_ema
        self.ddim_sample = ddim_sample
        self.ddim_sample_step = ddim_sample_step
        self.save_input = save_input
        self.openai_norm_keys = ["reference", "conditioning"]

    @rank_zero_only
    def log_local(self, images, global_step, current_epoch, batch_idx, img_idx, is_train):
        def save_image(img, path, openai=False):
            if self.rescale:
                if openai:
                    img = img * OPENAI_STD + OPENAI_MEAN
                    img = torch.clamp(img, 0, 1)
                else:
                    img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            img = img.permute(1, 2, 0).squeeze(-1)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(path)

        for k in images:
            dirpath = os.path.join(self.save_path, k)
            os.makedirs(dirpath, exist_ok=True)
            if is_train:
                # save grid images during training
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}.png"
                path = os.path.join(dirpath, filename)
                save_image(grid, path, k in self.openai_norm_keys)

            else:
                # save images separately during testing
                for idx in range(len(images[k])):
                    img = images[k][idx]
                    filename = f"{img_idx[idx]}.png"
                    path = os.path.join(dirpath, filename)
                    save_image(img, path, k in self.openai_norm_keys)

    def log_img(self, pl_module, batch, batch_idx):
        is_train = pl_module.training
        if (hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.sample_num > 0) or not is_train:
            if is_train:
                pl_module.eval()

            batch['sample'] = is_train
            images, index = pl_module.log_images(
                batch = batch,
                N = self.sample_num,
                sample = self.guidance_scale == 1.,
                unconditional_guidance_scale = self.guidance_scale,
                unconditional_guidance_label = self.guidance_label,
                use_ema_scope = self.use_ema,
                ddim_steps = self.ddim_sample_step if is_train or self.ddim_sample else None,
                return_inputs = self.save_input,
            )

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp and not k in self.openai_norm_keys:
                        images[k] = torch.clamp(images[k], -1., 1.)
                if len(images[k].shape) == 3:
                    images[k] = images[k].unsqueeze(0)

            self.log_local(images, pl_module.global_step, pl_module.current_epoch, batch_idx, index, is_train)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if not self.disabled and pl_module.global_step > 0 and self.check_frequency(check_idx):
            self.log_img(pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx)

@rank_zero_only
def setup_callbacks(opt, device_num=1, train=False):
    callbacks = [ImageLogger(
        batch_frequency  = default(opt, 'save_freq_step', 1),
        save_path        = opt.sample_path if train else opt.test_path,
        sample_num       = min(maxm_sample_size, default(opt, 'sample_num', opt.batch_size)),
        resume           = default(opt, 'resume', False),
        check_memory_use = default(opt, 'check_memory_use', False),
        guidance_scale   = opt.guidance_scale,
        guidance_label   = opt.guidance_label,
        use_ema          = opt.use_ema,
        ddim_sample      = opt.ddim,
        save_input       = default(opt, 'save_input', True),
    )]

    if train:
        batch_per_epoch = opt.data_size // (device_num * opt.batch_size)

        callbacks += [CustomCheckpoint(opt)]
        callbacks += [ConsoleLogger(opt.print_freq, opt.ckpt_path, batch_per_epoch=batch_per_epoch)]
    return callbacks