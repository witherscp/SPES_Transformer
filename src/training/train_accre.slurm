#!/bin/bash
#SBATCH --mail-user=christopher.p.withers@vanderbilt.edu
#SBATCH --mail-type=ALL

#SBATCH --account=accre_guests_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --output=train_slurm.out


repo_dir=/home/withercp/SPES_Transformer

source ${repo_dir}/torch/bin/activate
python train.py -c 1
