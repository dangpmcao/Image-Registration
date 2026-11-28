#!/usr/bin/env bash
#SBATCH --job-name=registration-download
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6%3

set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
root="$1"
exec bash "$root/Image-Registration/scripts/slurm/download_task.sh" "$root" "$SLURM_ARRAY_TASK_ID"
