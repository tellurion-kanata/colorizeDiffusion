import logger
import gc

from tqdm import tqdm
from options import Options
from util import load_config
from data.dataloader import create_dataloader
from sgm.util import instantiate_from_config, default

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


if __name__ == '__main__':
    opt, configs, device_num = get_configurations()

    # setup model and data loader
    model = instantiate_from_config(configs.model)
    dataloader, data_size = create_dataloader(opt, configs.dataloader, device_num)
    if opt.resume:
        model.init_from_ckpt(opt.load_checkpoint, ignore_keys=opt.ignore_keys, make_it_fit=opt.fitting_model)

    # setup huggingface accelerator
    projection_config = ProjectConfiguration(
        project_dir = opt.ckpt_path,
        automatic_checkpoint_naming = True,
        total_limit = 3,
    )
    accelerate = Accelerator(
        mixed_precision = opt.precision,
        log_with = "tensorboard",
        project_config = projection_config,
    )
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=opt.learning_rate)
    model, optimizer, dataloader = accelerate.prepare(model, optimizer, dataloader)

    # setup loggers
    # TODO: Check if there are alternative methods in huggingface libraries
    vars_opt = vars(opt)
    batch_per_epoch = int(data_size // opt.batch_size // device_num)
    ckpt_callback = logger.CustomCheckpoint(**vars_opt)
    if accelerate.is_local_main_process:
        vis_logger = logger.ImageLogger(**vars_opt)
        cli_logger = logger.ConsoleLogger(batch_per_epoch=batch_per_epoch, **vars_opt)
        pbar_epoch = tqdm(initial=opt.start_epoch, total=opt.epoch, desc="Training process")

    # start training
    global_step = 0
    model.training = True
    model.on_train_start()
    ckpt_callback.on_train_start(accelerate, model)
    accelerate.init_trackers(opt.name)
    for epoch in range(opt.start_epoch, opt.epoch):
        if accelerate.is_local_main_process:
            pbar_iter = tqdm(total=len(dataloader), desc=f"Current epoch {epoch}, process")

        for idx, batch in enumerate(dataloader):
            # forward and backward
            loss = model.training_step(batch, idx)
            accelerate.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # batch end callbacks
            loss_dict = {"loss": loss.item()}
            accelerate.log(loss_dict, step=global_step)
            model.on_train_batch_end()
            ckpt_callback.on_train_batch_end(accelerate, model, global_step, idx)

            if accelerate.is_local_main_process:
                training_state = {
                    "max_epoch": opt.epoch,
                    "learning_rate": opt.learning_rate,
                    "loss_dict": loss_dict,
                    "batch": batch,
                    "batch_idx": idx,
                    "global_step": global_step,
                    "current_epoch": epoch,
                }

                vis_logger.on_train_batch_end(model, **training_state)
                cli_logger.on_train_batch_end(**training_state)
                pbar_iter.set_postfix(loss_dict)
                pbar_iter.update(1)
                del training_state

            global_step += 1
            del loss_dict

        # epoch end callbacks
        ckpt_callback.on_train_epoch_end(accelerate, model, epoch)
        if accelerate.is_local_main_process:
            cli_logger.on_train_epoch_end(epoch, global_step, opt.learning_rate)
            pbar_epoch.update(1)
            pbar_epoch.close()
        gc.collect()

    accelerate.end_training()
    if accelerate.is_local_main_process:
        pbar_epoch.close()