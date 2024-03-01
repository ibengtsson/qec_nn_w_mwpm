#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 0-00:01:00		# days-hours:minutes:seconds
#SBATCH -J mwpm_ls_training
#SBATCH -o ../../alvis_out/mwpm_ls_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

module purge
source ~/bash_scripts/load_env.sh


python3 ../scripts/edge_ls.py
