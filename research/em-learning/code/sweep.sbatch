#!/bin/bash
#SBATCH --job-name=bpem
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/3164542/bp-em-diffusion/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/bp-em-diffusion/logs/%x_%A_%a.err
#
# EM-for-BP sweep, one array task per shard.
#
#   sbatch --array=0-59 --export=ALL,EXPERIMENT=identifiability,NSHARDS=60 sweep.sbatch
#
# Single-threaded on purpose: the linear algebra here is small (801x801 at most), so
# thread parallelism inside a task buys little, while one CPU per task lets many array
# tasks pack onto a node and finish sooner in wall-clock terms.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/3164542/bp-em-diffusion/.venv/bin/python
ROOT=/home/3164542/bp-em-diffusion
OUT=$ROOT/outputs/$EXPERIMENT

cd "$ROOT/code"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}"
echo "experiment=$EXPERIMENT"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"

srun "$PYTHON" run_sweep.py \
    --experiment "$EXPERIMENT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --nshards "$NSHARDS" \
    --out "$OUT"

echo "end=$(date -Is)"
