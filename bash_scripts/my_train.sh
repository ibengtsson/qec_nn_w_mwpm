#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00		# days-hours:minutes:seconds
#SBATCH -J mwpm_ls_training
#SBATCH -o ../../alvis_out/mwpm_ls_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

# load modules and environment
module purge
# source ~/scripts/load_env.sh

# use default config if no file is provided as an input
if [ $# -eq 0 ]
then
    echo "Using using default config file"
    config="../configs/ls_config.yaml"
else
    echo "Using config file: $1"
    config=$1
fi

apptainer exec ~/PyG.sif python ../scripts/train.py -c $config --save
# python3 ../scripts/train.py -c $config --save
