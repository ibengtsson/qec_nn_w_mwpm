#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 0-06:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J inference
#SBATCH -o ../job_outputs/inference%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A100:1 # GPUs 64GB of RAM; cost factor 1.0

# load modules and environment
module purge

# files
FILES=($(find ../training_outputs/report/add/ -type f))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# send script
apptainer exec ~/containers/torch/PyG_new.sif python ../scripts/inference_array.py -f $FILE
