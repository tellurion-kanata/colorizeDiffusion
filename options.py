import os
import argparse
import shutil
import json

from datetime import datetime
from refnet.sampling import get_sampler_list, get_noise_schedulers

def list_of_bools(value):
    try:
        value = json.loads(value)
        if not isinstance(value, list):
            raise ValueError
        if not all(isinstance(x, bool) for x in value):
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid list of bools")

def list_of_floats(value):
    try:
        value = json.loads(value)
        if not isinstance(value, list):
            raise ValueError
        for sub_list in value:
            if not all(isinstance(x, float) for x in sub_list):
                raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid list of list of floats")

class Options():
    def __init__(self, eval=False):
        self.initialize(eval)
        self.modify_options()


    def initialize(self, eval=False):
        self.eval = eval

        self.parser = argparse.ArgumentParser()
        # overall options
        self.parser.add_argument('--name', '-n', default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                 help='Project name under the checkpoints file')
        self.parser.add_argument('--dataroot', '-d', type=str, required=True,
                                 help='Dataset path')
        self.parser.add_argument('--batch_size', '-bs', default=32, type=int,
                                 help='Number of batch size')
        self.parser.add_argument('--pretrained', '-pt', type=str, default=None,
                                 help='Path of pre-trained model weights')
        self.parser.add_argument('--num_threads', '-nt', type=int, default=0,
                                 help='Number of threads when reading data')
        self.parser.add_argument('--save_path', type=str, default='./checkpoints',
                                 help='Trained models save path')
        self.parser.add_argument('--config_file', '-cfg', type=str, default=None,
                                 help='Model config file path. Default path is "[ckpt_path]/model_config.yaml" when testing')
        self.parser.add_argument('--guidance_scale', '-gs', type=float, default=1.0,
                                 help='Unconditional guidance scale for denoise sampling.')
        self.parser.add_argument('--guidance_label', '-gl', type=str, default='reference', choices=['sketch', 'reference'],
                                 help='Label used for unconditional guidance.')
        self.parser.add_argument('--use_ema', action='store_true',
                                 help='Use ema weights during sampling.')
        self.parser.add_argument('--steps', '-s', type=int, default=20,
                                 help='Denoising step')
        self.parser.add_argument('--seed', '-sd', type=int, default=None,
                                 help='Initialize global seed.')
        self.parser.add_argument('--ignore_keys', '-ik', type=str, default=[], nargs='*',
                                 help="Ignore keys when initialize from checkpoint.")
        self.parser.add_argument('--sampler', type=str, default='diffuser_dpm', choices=get_sampler_list(),
                                 help="Sampler used to generate images.")
        self.parser.add_argument('--scheduler', type=str, default='Automatic', choices=get_noise_schedulers(),
                                 help="Sampler used to generate images.")
        self.parser.add_argument('--load_logging', '-log', action='store_true',
                                 help="Logging missing and unexpected parameters when loading from a checkpoint.")

    def modify_options(self):
        if not self.eval:
            self.add_training_options()
        else:
            self.add_testing_options()

    def add_training_options(self):
        self.parser.add_argument('--fitting_model', '-fm', action='store_true',
                                 help='Fit the checkpoint states to the new model.')

        # training options
        self.parser.add_argument('--load_checkpoint', '-ckpt', type=str, default=None,
                                 help='Checkpoint to load.')
        self.parser.add_argument('--learning_rate', '-lr', default=1e-5, type=float,
                                 help="Learning rate")
        self.parser.add_argument('--dynamic_lr', '-dlr', action='store_true',
                                 help="Activate to adjust the learning rate based on the batch size and the number of devices.")
        self.parser.add_argument('--precision', choices=["bf16", "fp16", "fp32"], default="fp16",
                                 help='Precision used in training.')
        self.parser.add_argument('--accumulate_batches', '-ab', default=1, type=int,
                                 help='Number of accumulate batches')
        self.parser.add_argument('--epoch', type=int, default=10,
                                 help='Total training epoch')
        self.parser.add_argument('--start_epoch', '-ste', type=int, default=0,
                                 help='# of start epoch')

        # logging and checkpoint options
        self.parser.add_argument('--sample_num', '-sn', default=None, type=int,
                                 help='Number of batch size for sampling')
        self.parser.add_argument('--log_img_step', '-lis', type=int, default=None,
                                 help='Sample image every # steps')
        self.parser.add_argument('--save_first_step', action='store_true',
                                 help='Save the model in the beginning of a training')
        self.parser.add_argument('--not_save_first_step_epoch', action='store_true',
                                 help='Save the model in the begining of each epoch')
        self.parser.add_argument('--save_freq', '-sf', type=int, default=1,
                                 help='Saving network states per # epochs')
        self.parser.add_argument('--save_freq_step', '-sfs', type=int, default=5000,
                                 help='Saving latest network states per # steps')
        self.parser.add_argument('--start_save_ep', '-sts', type=int, default=0,
                                 help='Start to save model each epoch after training # epochs')
        self.parser.add_argument('--top_k', type=int, default=4,
                                 help='Save latest #-checkpoints')
        self.parser.add_argument('--print_freq', '-pf', type=int, default=1000,
                                 help='Print training states per iterations')
        self.parser.add_argument('--not_save_weight_only', '-st', action='store_true',
                                 help='Save training states when activated')

        # check memory use option
        self.parser.add_argument('--check_memory_use', '-cmu', action='store_true')

    def add_testing_options(self):
        self.parser.add_argument('--validation', '-val', action='store_true',
                                 help='Validation mode or testing mode')
        self.parser.add_argument('--eval_load_size', '-ls', type=int, default=None,
                                 help='Loading image size during testing, set batch size to 1 when activate this option')
        self.parser.add_argument('--save_input', '-si', action='store_true',
                                 help='Save input images during testing')
        self.parser.add_argument('--interpolate_positional_embedding', '-ipe', action='store_true',
                                 help='Interpolate the positional embedding used in the ViT')

    def dirsetting(self, opt):
        opt.ckpt_path = os.path.join(opt.save_path, opt.name)
        opt.model_config_file = os.path.join(opt.ckpt_path, 'model_config.yaml')
        os.makedirs(opt.ckpt_path, exist_ok=True)

        if opt.config_file is not None:
            shutil.copy(opt.config_file, os.path.join(opt.ckpt_path, 'model_config.yaml'))

    def parse(self, opt):
        if opt.dataroot[-1] == '\\':
            opt.dataroot = opt.dataroot[:-1]

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<40}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        mode = 'at' if not self.eval else 'wt'
        file_name = os.path.join(opt.ckpt_path, f'{opt.mode}_opt.txt')
        with open(file_name, mode) as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def get_options(self):
        opt = self.parser.parse_args()
        opt.eval = self.eval

        self.dirsetting(opt)
        self.parse(opt)

        return opt