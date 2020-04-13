#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# set max wallclock time
#SBATCH --time=12:00:00
# set name of job
#SBATCH --job-name=fast_free
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=amartya.sanyal@cs.ox.ac.uk
# how many gpus you want to use
#SBATCH --gres=gpu:1
# which gpus you want to use
#SBATCH --constraint="gpu_sku:P100|gpu_sku:V100"
# during reservation period (usually before deadlines), we can use following for faster gpu allocation
#SBATCH --reservation=tvg_042020
#SBATCH --array=1-4
source ~/.bashrc
module load gpu/cuda/9.0.176 gpu/cudnn/7.3.1__cuda-9.0



conda activate /home/shug5742/miniconda3/envs/pytorch

# Pass the bern noise level
PARRAY=(8 10 16 20)
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID`]}
python train_fgsm.py --epsilon  $p1 --early-stop --epochs 350
