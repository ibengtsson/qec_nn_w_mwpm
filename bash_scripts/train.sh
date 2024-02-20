#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J nn_training
#SBATCH -o ../job_outputs/nn_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

# load modules and environment
module purge
source ~/scripts/load_env.sh

# send script
config=$1
python3 ../scripts/train.py -c $config