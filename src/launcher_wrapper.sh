#!/bin/sh
cat <<EoF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=nestor,ig-88
#SBATCH --output=slurm_logs/$1.out
#SBATCH --error=slurm_logs/$1.err
#SBATCH --account=kira-lab

source /nethome/hmaheshwari7/.bashrc
conda deactivate
conda activate mmsemienv

set -x

echo "Launching training"
srun python main_ddp.py --cfg_file $2
EoF