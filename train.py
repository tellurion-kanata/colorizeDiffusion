import logger
import psutil
import os.path as osp

from tqdm import tqdm
from options import Options
from util import load_config
from data.dataloader import create_dataloader
from refnet.util import instantiate_from_config, default

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


def get_configurations():
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = "train"
    device_num = torch.cuda.device_count()
    configs = load_config(default(opt.config_file, opt.model_config_file))

    if opt.dynamic_lr:
        base_lr = configs.model.base_learning_rate
        opt.learning_rate = base_lr * opt.batch_size * opt.acumulate_batch_size * len(opt.gpus)
    parser.print_options(opt)
    return opt, configs, device_num


def get_system_memory_usage_gb():
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3), memory.used / memory.total * 100.


if __name__ == '__main__':
    opt, configs, device_num = get_configurations()

    # setup model and data loader
    dataloader, data_size = create_dataloader(opt, configs.dataloader, device_num)
    model = instantiate_from_config(configs.model)
    params = model.get_trainable_params()
    tqdm.write(f"Total parameter number: {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f} B, "
               f"trainable parameter number: {sum(p.numel() for p in params) / 1000 ** 3:.2f} B !!!")
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate)
    if opt.pretrained is not None:
        model.init_from_ckpt(
            opt.pretrained,
            make_it_fit = opt.fitting_model,
            ignore_keys = opt.ignore_keys,
            logging = opt.load_logging
        )

    # setup huggingface accelerator
    projection_config = ProjectConfiguration(project_dir=opt.ckpt_path)
    accelerator = Accelerator(
        mixed_precision = opt.precision if opt.precision != "fp32" else None,
        log_with = "tensorboard",
        project_config = projection_config,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # resume training
    if opt.load_checkpoint:
        accelerator.load_state(opt.load_checkpoint)

    # setup loggers
    # TODO: Check if there are alternative methods in huggingface libraries
    vars_opt = vars(opt)
    batch_per_epoch = int(data_size // opt.batch_size // device_num)
    ckpt_callback = logger.CustomCheckpoint(**vars_opt)
    if accelerator.is_main_process:
        vis_logger = logger.ImageLogger(**vars_opt)
        cli_logger = logger.ConsoleLogger(batch_per_epoch=batch_per_epoch, **vars_opt)
        pbar_epoch = tqdm(initial=opt.start_epoch, total=opt.epoch, desc="Training process")

    # start training
    global_step = 0
    model.training = True
    model.on_train_start()
    ckpt_callback.on_train_start(accelerator)
    accelerator.init_trackers(opt.name)
    for epoch in range(opt.start_epoch, opt.epoch):
        if accelerator.is_main_process:
            pbar_iter = tqdm(total=len(dataloader), desc=f"Current epoch {epoch}, process")

        # Epoch starts
        for idx, batch in enumerate(dataloader):
            # forward and backward
            loss = model.training_step(batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # batch end callbacks
            model.on_train_batch_end()
            loss_dict = {"loss": accelerator.gather(loss.repeat(opt.batch_size)).mean().item()}

            if accelerator.sync_gradients:
                ckpt_callback.on_train_batch_end(accelerator, global_step, idx)

                if accelerator.is_main_process:
                    # check cpu memory usage for deepspeed ZeRO
                    # ram_used, ram_usage = get_system_memory_usage_gb()
                    # logging_dict = {
                    #     "cpu_memory_used (GB)": ram_used,
                    #     "cpu_memory_usage (%)": ram_usage,
                    #     "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024 ** 2),
                    # }
                    # logging_dict.update(loss_dict)

                    logging_dict = loss_dict
                    training_state = {
                        "max_epoch": opt.epoch,
                        "learning_rate": opt.learning_rate,
                        "loss_dict": loss_dict,
                        "batch": batch,
                        "batch_idx": idx,
                        "global_step": global_step,
                        "current_epoch": epoch,
                    }

                    accelerator.log(logging_dict, step=global_step)
                    vis_logger.on_train_batch_end(model, **training_state)
                    cli_logger.on_train_batch_end(**training_state)
                    pbar_iter.set_postfix(loss_dict)
                    pbar_iter.update(1)
                    del logging_dict, training_state

            global_step += 1
            del loss_dict

        # epoch end callbacks
        model.on_train_epoch_end()

        if accelerator.sync_gradients:
            ckpt_callback.on_train_epoch_end(accelerator, epoch)
            
            if accelerator.is_main_process:
                cli_logger.on_train_epoch_end(epoch, global_step, opt.learning_rate)
                pbar_epoch.update(1)
                pbar_iter.close()

    accelerator.save_state(osp.join(opt.ckpt_path, "final"))
    accelerator.end_training()
    if accelerator.is_main_process:
        pbar_epoch.close()