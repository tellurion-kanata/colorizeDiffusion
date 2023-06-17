import yaml
import torch
import pytorch_lightning as pl

from data import CustomDataLoader
from options import Options
from logger import setup_callbacks
from utils import default, instantiate_from_config


if __name__ == '__main__':
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = 'train'

    # Load config file
    try:
        with open(opt.config_file, 'r') as f:
            configs = yaml.safe_load(f.read())
        cfg_path = opt.config_file
    except:
        print(f"{opt.config_file} does not exist, try to load config file in the checkpoint file")
        try:
            with open(opt.model_config_path, 'r') as f:
                configs = yaml.safe_load(f.read())
            cfg_path = opt.model_config_path
        except:
            raise IOError("Failed in loading model configs, please check the training settings.")
    print(f"Loaded model config from {cfg_path}")

    # Get training dataset information and log training options
    device_num = torch.cuda.device_count() if opt.device == "auto" else len(opt.device)
    dataset = CustomDataLoader(opt, configs['dataloader'], device_num)
    opt.data_size = dataset.get_data_size()
    if opt.dynamic_lr:
        base_lr = configs['model']['base_learning_rate']
        opt.learning_rate = base_lr * opt.batch_size * opt.acumulate_batch_size * len(opt.device)
    parser.print_options(opt, phase='train')

    # Todo: Check if Automatic mixed precision (AMP) cannot be used jointly with openai-checkpoint
    # if opt.use_amp and 'unet_config' in configs['model']['params']:
    #     configs['model']['params']['unet_config']['params']['use_checkpoint'] = False

    model = instantiate_from_config(configs['model'])
    model.lr = opt.learning_rate
    
    # Define lightning trainer
    callbacks = setup_callbacks(opt, device_num, True)
    trainer = pl.Trainer(
        benchmark               = True,
        max_epochs              = default(opt, 'niter'),
        devices                 = opt.device,
        log_every_n_steps       = default(opt, 'print_freq'),
        accumulate_grad_batches = default(opt, 'accumulate_batches', 1),
        default_root_dir        = opt.ckpt_path,
        callbacks               = callbacks,
        accelerator             = opt.accelerator,
        precision               = '32-true' if opt.not_use_amp else '16-mixed',
        enable_checkpointing    = False,
        strategy                = 'ddp' if device_num > 1 else 'auto',
    )

    # Resume w/ and w/o training states
    load_checkpoint = None
    if opt.resume:
        if opt.load_training_states:
            load_checkpoint = opt.load_checkpoint
        else:
            model.make_it_fit = opt.fitting_model
            model.init_from_ckpt(opt.load_checkpoint, ignore_keys=opt.ignore_keys, log_missing=True)

    # model = torch.compile(model)
    # start training
    trainer.fit(model, dataset, ckpt_path=load_checkpoint)