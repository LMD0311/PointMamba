


import datetime
import random

from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune

from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
import os
import argparse
from pathlib import Path
import numpy as np


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help='vote acc')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help='finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help='training modelnet from scratch')
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help='difficulty mode for shapenet')
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode

    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,
                                        args.exp_name,
                                        ('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
                                        )
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',
                                     args.exp_name,
                                     ('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
                                     )
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    # if args.use_gpu:
    #     torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs // args.world_size
            # config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // args.world_size
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs // args.world_size
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // args.world_size
    else:
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs
        if config.dataset.get('extra_val'):
            config.dataset.extra_val.others.bs = config.total_bs
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    if args.test:
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
