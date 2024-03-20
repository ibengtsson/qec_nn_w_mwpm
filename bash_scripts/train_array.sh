#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J mwpm_ls_training
#SBATCH -o ../../alvis_out/mwpm_ls_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

# purge any modules
module purge
apptainer exec ~/conda-cont.sif python ../scripts/train.py -c ../configs/array_configs/config_${SLURM_ARRAY_TASK_ID}.yaml --save
