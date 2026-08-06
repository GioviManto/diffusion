#!/bin/bash
#SBATCH --job-name=exp16
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_16 (sampling validation) on the Bocconi cluster.
#
# Unlike the templates in this directory, this one is filled in for a specific site:
# partition `medium_cpu`, account implicit, and the QOS caps that actually apply here
# (30 jobs submitted, 10 running per user) -- so array sizes above 30 will be rejected
# outright rather than queued.
#
#   Calibration (must run and be inspected FIRST -- see below):
#     sbatch --array=0-5 --export=ALL,MODE=calibrate bocconi_exp16.sbatch
#
#   Main run, one task per (size, seed) cell:
#     sbatch --array=0-15 --export=ALL,MODE=generate,NSTEPS=<from calibration> \
#            bocconi_exp16.sbatch
#
# WHY CALIBRATION COMES FIRST. The reference arm of this experiment is the *exact* score,
# and a smoke run showed it does not reproduce the true innovation kurtosis at small step
# counts: 1.78 at 60 steps and 2.32 at 120, against a true 3.0. That is the Euler-Maruyama
# discretization, not the score. Until the exact arm reproduces the truth, a comparison
# between arms measures the integrator. So: run MODE=calibrate, read off the step count at
# which the exact arm's kurtosis stops moving, and pass it as NSTEPS to the main run.
# Do not skip this and do not guess the number.
#
# Single-threaded on purpose: the per-task linear algebra is a 401x401 transition matrix,
# so thread parallelism buys little while one CPU per task lets many tasks pack onto a node.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=/home/3164542/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
CODE=$ROOT/research/nongaussian-bp

cd "$CODE"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-generate}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"
echo

case "$MODE" in

  calibrate)
    # One task per step count. Only the `exact` arm matters here, but the part runs all
    # four; the others are cheap by comparison and their trend is informative too.
    LADDER=(100 200 400 800 1600 3200)
    NS=${LADDER[$TASK]}
    OUT=$ROOT/outputs/exp_16/calibrate_steps${NS}
    echo "calibrating at n_steps=$NS -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only steps \
        --output-dir "$OUT" \
        --set steps_ladder="($NS,)" \
        --set n_generate=2000 \
        --set sizes="(2048,)" \
        --set n_seed=1
    ;;

  generate)
    # One task per (size, seed). Cells write disjoint output directories, so there is no
    # merge step beyond concatenating CSVs.
    SIZES=(32 128 512 2048)
    NSEED=4
    SI=$(( TASK / NSEED ))
    SEED=$(( TASK % NSEED ))
    SZ=${SIZES[$SI]}
    NSTEPS=${NSTEPS:?"NSTEPS must be set from the calibration run"}
    OUT=$ROOT/outputs/exp_16/gen_n${SZ}_seed${SEED}
    echo "generate: size=$SZ seed=$SEED n_steps=$NSTEPS -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate,pointwise \
        --output-dir "$OUT" \
        --set sizes="($SZ,)" \
        --set n_seed=1 \
        --set n_steps="$NSTEPS" \
        --set seed_offset="$SEED"
    ;;

  *)
    echo "unknown MODE=$MODE (expected calibrate or generate)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
