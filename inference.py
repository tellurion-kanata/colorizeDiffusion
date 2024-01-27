import logger

from tqdm import tqdm
from options import Options
from util import load_config
from data.dataloader import create_dataloader
from sgm.util import instantiate_from_config, default



def get_configurations():
    parser = Options(eval=True)
    opt = parser.get_options()
    opt.mode = "validation" if opt.validation else "test"
    opt.log_img_step = 1
    configs = load_config(default(opt.config_file, opt.model_config_file))
    parser.print_options(opt)
    return opt, configs


def modify_z_shape(params, eval_size, interpolate=False):
    ldm_image_size = params.image_size
    first_model_config = params.first_stage_config
    first_stage_image_size = first_model_config.params.ddconfig.resolution
    scale_factor = first_stage_image_size // ldm_image_size

    params.image_size = eval_size // scale_factor
    if interpolate:
        try:
            params.cond_stage_config.params.clip_config.scale_factor = eval_size / first_stage_image_size
        except:
            params.cond_stage_config.params.scale_factor = eval_size / first_stage_image_size
    return params


if __name__ == '__main__':
    opt, configs = get_configurations()
    model_params = configs.model.params
    if opt.eval_load_size is not None and 'image_size' in model_params:
        configs.model.params = modify_z_shape(model_params, opt.eval_load_size, opt.interpolate_positional_embedding)

    # setup model and data loader
    model = instantiate_from_config(configs.model).cuda().eval()
    model.switch_to_fp16()
    model.cond_stage_model = model.cond_stage_model.half()
    model.first_stage_model = model.first_stage_model.half()
    dataloader, data_size = create_dataloader(opt, configs.dataloader, 1, eval_load_size=opt.eval_load_size)
    model.init_from_ckpt(opt.pretrained, ignore_keys=opt.ignore_keys)

    vars_opt = vars(opt)
    vis_logger = logger.ImageLogger(**vars_opt)

    model.training = False
    for idx, batch in enumerate(tqdm(dataloader)):
        for key in batch.keys():
            batch[key] = batch[key].cuda()
        vis_logger.on_test_batch_end(model, batch, idx)