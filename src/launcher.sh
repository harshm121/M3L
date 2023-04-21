#!/bin/bash
#SBATCH --job-name=tfdepth
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 64
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=nestor,ig-88
#SBATCH --output=slurm_logs/tfdepth.out
#SBATCH --error=slurm_logs/tfdepth.err
#SBATCH --account=kira-lab

source /nethome/hmaheshwari7/.bashrc
conda deactivate
conda activate mmsemienv

set -x

echo "Launching training"
srun python main_ddp.py --cfg_file configs/rgbd/stanford_indoor/proposed/tokenfusion_sup14_predmaskedpseudosup95_cleanandmask_learntmask_depthonlymask_1-500_bs_32.yml --verbose iter
