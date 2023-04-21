import argparse
import wandb
import os
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from omegaconf.listconfig import ListConfig
import torch
import torch.distributed as dist
import numpy as np
import random

from semi_supervised.get_framework import get_framework


def setup(rank, port, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_args(args_cmd):
    args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
    args = OmegaConf.create(args_cfg)
    args.verbose = args_cmd.verbose
    args.rgb_root = args.data_root
    args.depth_root = args.data_root
    args.gt_root = args.data_root
    args.sup_train_source = os.path.join(args.train_source, f"train_labeled_1-{args.labeled_ratio}.txt")
    args.unsup_train_source = os.path.join(args.train_source, f"train_unlabeled_1-{args.labeled_ratio}.txt")
    args.num_train_imgs =  args.total_train_imgs // args.labeled_ratio
    args.num_unsup_imgs = args.total_train_imgs - args.num_train_imgs
    args.max_samples = max(args.num_train_imgs, args.num_unsup_imgs)
    args.niters_per_epoch = args.max_samples // args.batch_size
    args.wandb_project = f"{args.data}_{args.modalities}_{args.ssl_framework}_{args.seg_model}_{args.base_model}_{args.image_height}_{args.image_width}_{args.labeled_ratio}"
    if not "optim" in args:
        args.optim = 'sgd'
    if not "freeze_bn" in args:
        args.freeze_bn = False
    if "run_suffix" in args:
        run_suffix = "_" + args.run_suffix
    else:
        run_suffix = ""
    args.run_name = f"lr={args.base_lr}_ssl={args.ssl_framework}_batchsize={args.batch_size}_GPU={args.world_size}{run_suffix}"
    args.snapshot_dir = f'{args.root_dir}/snapshots/{args.data}/{args.ssl_framework}/1-{args.labeled_ratio}/{args.seg_model}/{args.base_model}/{args.run_name}'
    os.makedirs(args.snapshot_dir, exist_ok = True)

    if not type(args.weight_decay) == ListConfig:
        args.weight_decay = [args.weight_decay for _ in range(len(args.base_lr))]
    return args

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(rank, args, port):
    set_random_seed(0, deterministic=False)
    setup(rank, port, args.world_size)
    args.rank = rank
    framework = get_framework(args)
    print(f"Starting training on rank = {rank} ...")
    if rank == 0:
        wandblogger = wandb.init(project=args.wandb_project)
        wandb.run.name = args.run_name
        framework.train(wandblogger, verbose=args.verbose)
    else:
        framework.train(None, verbose=args.verbose)

if __name__ == '__main__':
    print("Launching...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file")
    parser.add_argument('--verbose', type=str, help="Either 'iter' or 'epoch'", default='epoch')
    args_cmd = parser.parse_args()
    args = get_args(args_cmd)
    port = str(int(random.uniform(12040, 12080)))
    print("Spawning...")
    mp.spawn(main, args = (args, port), nprocs = args.world_size)