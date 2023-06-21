import yaml
import torch
import pytorch_lightning as pl

from data import CustomDataLoader
from options import Options
from logger import setup_callbacks
from utils import default, instantiate_from_config


def modify_z_shape(params, eval_size):
    ldm_image_size = params['image_size']
    first_model_config = params['first_stage_config']
    first_stage_image_size = first_model_config['params']['ddconfig']['resolution']
    scale_factor = first_stage_image_size // ldm_image_size

    params['image_size'] = eval_size // scale_factor
    try:
        params['cond_stage_config']['params']['clip_config']['scale_factor'] = eval_size / first_stage_image_size
    except:
        params['cond_stage_config']['params']['scale_factor'] = eval_size / first_stage_image_size
    return params


if __name__ == '__main__':
    parser = Options(eval=True)
    opt = parser.get_options()

    pl.seed_everything(opt.seed)
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

    device_num = torch.cuda.device_count() if opt.device == "auto" else len(opt.device)
    dataset = CustomDataLoader(
        opt             = opt,
        cfg             = configs['dataloader'],
        device_num      = device_num,
        eval_load_size  = opt.eval_load_size,
        save_input      = opt.save_input
    )
    opt.data_size = dataset.get_data_size()
    parser.print_options(opt, phase='test')

    model_params = configs['model']['params']
    if opt.eval_load_size is not None and 'image_size' in model_params:
        configs['model']['params'] = modify_z_shape(model_params, opt.eval_load_size)

    model = instantiate_from_config(configs['model'])
    callbacks = setup_callbacks(opt)
    trainer = pl.Trainer(
        benchmark               = True,
        max_epochs              = default(opt, 'niter'),
        devices                 = opt.device,
        log_every_n_steps       = default(opt, 'print_freq'),
        accumulate_grad_batches = default(opt, 'accumulate_batches', 1),
        default_root_dir        = opt.ckpt_path,
        callbacks               = callbacks,
        accelerator             = opt.accelerator,
    )

    # model = torch.compile(model)
    model.init_from_ckpt(opt.load_checkpoint, ignore_keys=opt.ignore_keys)
    trainer.test(model, dataset)