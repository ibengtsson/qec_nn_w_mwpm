#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J mwpm_nn_training
#SBATCH -o ../job_outputs/mwpm_nn_training_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1

# purge any modules
module purge
apptainer exec ~/containers/torch/PyG_new.sif python ../scripts/nested_train.py -c ../configs/nested/config_${SLURM_ARRAY_TASK_ID}.yaml --save
