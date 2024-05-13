#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-353 -p alvis
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J inference
#SBATCH -o ../job_outputs/inference%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# load modules and environment
module purge

# files
FILES=($(find ../saved_models/report/ -type f))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# send script
apptainer exec ~/containers/torch/PyG_new.sif python ../scripts/inference.py -f $FILE
