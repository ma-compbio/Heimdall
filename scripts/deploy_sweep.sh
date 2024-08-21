#!/bin/bash
#SBATCH --job-name sweep
#SBATCH --time 0-12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem=32Gb
#SBATCH -p ma2-gpu
##SBATCH -w compute-0-37
#SBATCH -w compute-4-26
#SBATCH --ntasks-per-node=1
#SBATCH --output=vscode_out.log

## usage: sbatch deploy_sweep.sh --experiment-name pancreas --project-name Pancreas-Celltype-Classification

module load cuda-12.2

source /home/nzh/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

CMD=$(python3 scripts/create_sweep.py "$@")

echo $CMD
eval $CMD
