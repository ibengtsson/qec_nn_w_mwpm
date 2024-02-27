#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J mwpm_nn_training
#SBATCH -o ../job_outputs/mwpm_nn_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

# load modules and environment
module purge
source ~/scripts/load_env.sh

# use default config if no file is provided as an input
if [ $# -eq 0 ]
then
    echo "Using using default config file"
    config="../configs/default_config.yaml"
else
    echo "Using config file: $1"
    config=$1
fi

python3 ../scripts/train.py -c $config --save