import os
import argparse
import shutil

from pytorch_lightning.utilities import rank_zero_only

class Options():
    def __init__(self, eval=False):
        self.initialize(eval)
        self.modify_options()


    def initialize(self, eval=False):
        self.eval = eval

        self.parser = argparse.ArgumentParser()
        # overall options
        self.parser.add_argument('--name', '-n', required=True,
                                 help='Project name under the checkpoints file')
        self.parser.add_argument('--dataroot', '-d', type=str, required=True,
                                 help='Dataset path')
        self.parser.add_argument('--device', type=str, default='auto',
                                 help='gpu ids:  e.g. 0 | 0,1 | 0,2 | -1 for cpu')
        self.parser.add_argument('--accelerator', '-ac', type=str, default='gpu',
                                 help='Accelerator used for running the models')
        self.parser.add_argument('--batch_size', '-bs', default=32, type=int,
                                 help='Number of batch size')
        self.parser.add_argument('--load_checkpoint', '-ckpt', type=str, default=None,
                                 help='Checkpoint to load. Default is \'latest.ckpt\' under the checkpoint directory.')
        self.parser.add_argument('--num_threads', '-nt', type=int, default=0,
                                 help='Number of threads when reading data')
        self.parser.add_argument('--save_path', '-s', type=str, default='./checkpoints',
                                 help='Trained models save path')
        self.parser.add_argument('--config_file', '-cfg', type=str, default=None,
                                 help='Model config file path. Default path is "[ckpt_path]/model_config.yaml" when testing')
        self.parser.add_argument('--guidance_scale', '-gs', type=float, default=1.0,
                                 help='Unconditional guidance scale for denoise sampling.')
        self.parser.add_argument('--guidance_label', '-gl', type=str, default='reference', choices=['sketch', 'reference'],
                                 help='Label used for unconditional guidance.')
        self.parser.add_argument('--use_ema', action='store_true',
                                 help='Use ema weights during sampling.')
        self.parser.add_argument('--ddpm', action='store_true',
                                 help='Use DDIM sapmler during sampling.')
        self.parser.add_argument('--ddim_step', type=int, default=200,
                                 help='DDIM sampler step')
        self.parser.add_argument('--seed', type=int, default=None,
                                 help='Initialize global seed.')
        self.parser.add_argument('--ignore_keys', '-ik', type=str, default=[], nargs='*',
                                 help="Ignore keys when initialize from checkpoint.")

    def modify_options(self):
        if not self.eval:
            self.add_training_options()
        else:
            self.add_testing_options()

    def add_training_options(self):
        self.parser.add_argument('--resume', action='store_true',
                                 help='Resume training')
        self.parser.add_argument('--load_training_states', '-lt', action='store_true',
                                 help='Load training information when resume. (# using lightning resume)')
        self.parser.add_argument('--fitting_model', '-fm', action='store_true',
                                 help='Fit the checkpoint states to the new model.')

        # training options
        self.parser.add_argument('--learning_rate', '-lr', default=1e-5, type=float,
                                 help="Learning rate")
        self.parser.add_argument('--dynamic_lr', '-dlr', action='store_true',
                                 help="Activate to adjust the learning rate based on the batch size and the number of devices.")
        self.parser.add_argument('--not_use_amp', action='store_true',
                                 help='Activate automatic mixed precision training, not available for latent diffusion training.')
        self.parser.add_argument('--acumulate_batch_size', '-ab', default=1, type=int,
                                 help='Number of accumulate batches')
        self.parser.add_argument('--sample_num', '-sn', default=None, type=int,
                                 help='Number of batch size for sampling')
        self.parser.add_argument('--niter', type=int, default=5,
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0,
                                 help='# of iter to linearly decay learning rate to zero')

        # image pre-processing and training states output
        self.parser.add_argument('--save_first_step', action='store_true',
                                 help='Save the model in the beginning of a training')
        self.parser.add_argument('--save_freq', '-sf', type=int, default=1,
                                 help='Saving network states per # epochs')
        self.parser.add_argument('--save_freq_step', '-sfs', type=int, default=10000,
                                 help='Saving latest network states per # steps')
        self.parser.add_argument('--start_save_ep', '-sts', type=int, default=0,
                                 help='Start to save model each epoch after training # epochs')
        self.parser.add_argument('--top_k', type=int, default=5,
                                 help='Save latest #-checkpoints')
        self.parser.add_argument('--print_freq', '-pf', type=int, default=1000,
                                 help='Print training states per iterations')
        self.parser.add_argument('--not_save_weight_only', '-st', action='store_true',
                                 help='Save training states when activated')

        # check memory use option for image logger
        self.parser.add_argument('--check_memory_use', '-cmu', action='store_true')

    def add_testing_options(self):
        self.parser.add_argument('--eval_load_size', '-ls', type=int, default=None,
                                 help='Loading image size during testing, set batch size to 1 when activate this option')
        self.parser.add_argument('--save_input', '-si', action='store_true',
                                 help='Save input images during testing')

        self.parser.add_argument('--not_sample_original_cond', '-nso', action='store_true',
                                 help='Sampling using original conditions')
        self.parser.add_argument('--target_scale', '-ts', type=float, default=[], nargs='*',
                                 help='Target scale for prompt-based manipulation')
        self.parser.add_argument('--control_prompt', '-ctl', type=str, default=[], nargs='*',
                                 help='Text prompt used to compute the position weight matrix')
        self.parser.add_argument('--target_prompt', '-txt', type=str, default=[], nargs='*',
                                 help='Text prompt used for prompt-based manipulation')
        self.parser.add_argument('--locally', '-loc', action='store_true')
        self.parser.add_argument('--thresholds', '-thres', type=float, default=[0.5, 0.55, 0.65, 0.95], nargs='*')

    def dirsetting(self, opt):
        def makedir(paths):
            for p in paths:
                if not os.path.exists(p):
                    os.mkdir(p)

        opt.ckpt_path = os.path.join(opt.save_path, opt.name)
        opt.sample_path = os.path.join(opt.ckpt_path, 'train')
        opt.test_path = os.path.join(opt.ckpt_path, 'test')
        opt.model_config_path = os.path.join(opt.ckpt_path, 'model_config.yaml')
        makedir([opt.save_path, opt.ckpt_path])

        if opt.load_checkpoint is None:
            opt.load_checkpoint = os.path.join(opt.ckpt_path, 'latest.ckpt')
        if opt.config_file is not None:
            shutil.copy(opt.config_file, os.path.join(opt.ckpt_path, 'model_config.yaml'))

    def parse(self, opt):
        if opt.dataroot[-1] == '\\':
            opt.dataroot = opt.dataroot[:-1]
        str_ids = opt.device.split(',')

        if opt.device != "auto":
            opt.device = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.device.append(id)

    @rank_zero_only
    def print_options(self, opt, phase):
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
        file_name = os.path.join(opt.ckpt_path, '{}_opt.txt'.format(phase))
        with open(file_name, mode) as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def get_options(self):
        opt = self.parser.parse_args()
        opt.eval = self.eval

        self.dirsetting(opt)
        self.parse(opt)

        return opt