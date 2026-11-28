#!/usr/bin/env bash
#SBATCH --job-name=dlir-sunnybrook
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=1-3%1

# Submit with --output pointing to an existing directory; see README.md.
set -euo pipefail
root="$1"
cd "$root/Image-Registration"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dlir-${SLURM_JOB_ID}"
# CUDA_VISIBLE_DEVICES is assigned by Slurm; do not overwrite it.
python_bin="${DLIR_PYTHON:-/home/dang.cpm/miniconda3/bin/python}"
"$python_bin" -u train.py --architecture dlir --dataset sunnybrook \
    --preset sunnybrook-multi --fold "$SLURM_ARRAY_TASK_ID" \
    --run-name "multi_fold${SLURM_ARRAY_TASK_ID}_seed42" \
    --steps-per-stage 10000 --device cuda "${@:2}"
