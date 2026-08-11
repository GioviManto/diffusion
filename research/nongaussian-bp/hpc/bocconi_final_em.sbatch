#!/bin/bash
#SBATCH --job-name=finalem
#SBATCH --partition=compute
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# The final EM-BP sweep set. Filled in for Bocconi: partition `compute`
# (3-day limit, cnode05-07), account implicit. QOS `normal` caps at 30 SUBMITTED
# and 10 RUNNING tasks per user, so no single submission may exceed 30 and the
# waves below are sized to stay under it together.
#
# Wave 1 (submit together, 21 tasks):
#   GATE=$(sbatch --parsable --partition=medium_cpu --time=00:45:00 \
#          --export=ALL,MODE=smoke hpc/bocconi_final_em.sbatch)
#   sbatch --dependency=afterok:$GATE --array=0-15%8 --cpus-per-task=4 --mem=12G \
#          --export=ALL,MODE=reps hpc/bocconi_final_em.sbatch
#   sbatch --dependency=afterok:$GATE --array=0-3%4 \
#          --export=ALL,MODE=recovery hpc/bocconi_final_em.sbatch
#
# Wave 2 (only once wave 1 has drained below ~12 submitted, 18 tasks):
#   sbatch --array=0-17%8 --export=ALL,MODE=shape hpc/bocconi_final_em.sbatch
#
# WHY THE GATE. Every downstream conclusion depends on two corrections made on
# 2026-08-10: the sign of the mixture rho gradient, and the returned-iterate
# alignment of the gradient route. A sweep launched against a stale checkout
# would burn days and produce numbers indistinguishable from the ones we are
# replacing. MODE=smoke runs the suite and refuses to let anything start if it
# fails, so a stale or half-synced tree cannot silently become a result.
#
# WHY CPU AND NOT GPU. The recursion carries a CuPy backend (src/backend.py) and
# does run on the A100 nodes, but this cluster has two of them against eight CPU
# nodes, and the QOS ceiling of 10 running tasks is what actually bounds this
# sweep. Twelve independent CPU tasks finish sooner than two faster GPU ones.
# The GPU path is validated separately by hpc/bocconi_gpu.sbatch, which runs the
# CPU/GPU parity suite before it is allowed to produce anything.

set -euo pipefail

ROOT=/home/3164542/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"
mkdir -p "$ROOT/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-smoke}
STAMP=$(date +%Y%m%d)
RUN_ROOT="outputs/final_em/${STAMP}"

echo "host=$(hostname) mode=$MODE task=$TASK cpus=${SLURM_CPUS_PER_TASK:-?} \
job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}"
$PYTHON -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

case "$MODE" in

smoke)
    # The gate. Nothing else is allowed to run if this fails.
    $PYTHON -m pytest tests/ -q
    $PYTHON experiments/exp_27_shape_convergence.py --quick \
        --output-dir "$RUN_ROOT/smoke/exp27"
    $PYTHON experiments/exp_06_em_parameter_recovery.py --quick \
        --only clean_raw_mle --output-dir "$RUN_ROOT/smoke/exp06"
    $PYTHON experiments/exp_07_em_vs_score_network.py --quick \
        --only sample_efficiency_val --output-dir "$RUN_ROOT/smoke/exp07"
    $PYTHON ../../tools/audit_em_bp_provenance.py
    echo "SMOKE OK"
    ;;

reps)
    # Headline sample-efficiency table at 16 independent training seeds, up from
    # the 6 the current table rests on. `sample_efficiency_val` is used rather
    # than `sample_efficiency` because it emits the validation-selected AND the
    # test-oracle column from the SAME fitted models -- so the honest protocol
    # and the sensitivity analysis cannot drift apart, and the networks are not
    # refitted twice.
    SEED="$TASK"
    OUT="$RUN_ROOT/reps/seed_${SEED}"
    $PYTHON experiments/exp_07_em_vs_score_network.py \
        --only sample_efficiency_val \
        --output-dir "$OUT" \
        --set "seed=$SEED" \
        --set 'sizes=(32,64,128,256,512,1024,2048,4096)' \
        --set net_steps=60000
    ;;

recovery)
    # Parameter recovery at replication high enough to settle the questions that
    # three replicates could not. Task 0 is the one that matters most: it is the
    # grid-free clean MLE against the grid-binned fit, which is what retired the
    # "discretisation floor" reading of the clean arm.
    case "$TASK" in
      0) $PYTHON experiments/exp_06_em_parameter_recovery.py --only clean_raw_mle \
             --set n_rep_raw=64 --output-dir "$RUN_ROOT/recovery/clean_raw" ;;
      1) $PYTHON experiments/exp_06_em_parameter_recovery.py --only clean_vs_noised \
             --set n_rep_clean=32 --output-dir "$RUN_ROOT/recovery/clean_vs_noised" ;;
      2) $PYTHON experiments/exp_06_em_parameter_recovery.py --only rate \
             --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,2048)' \
             --output-dir "$RUN_ROOT/recovery/rate" ;;
      3) $PYTHON experiments/exp_06_em_parameter_recovery.py \
             --only monotonicity,quantization \
             --set n_inits=16 --output-dir "$RUN_ROOT/recovery/monotonicity" ;;
      *) echo "no recovery cell $TASK" >&2; exit 1 ;;
    esac
    ;;

shape)
    # The convergence sweep. One task per (N, C, design) cell, 4 seeds and 800
    # updates inside each, because "is 120 iterations enough" has a different
    # answer in every cell and the claim has been revised three times for want of
    # this measurement.
    SIZES=(512 2048)
    COMPS=(4 8 16)
    DESIGNS=(low uniform high)
    si=$(( TASK / 9 ))
    ci=$(( (TASK % 9) / 3 ))
    di=$(( TASK % 3 ))
    N=${SIZES[$si]}; C=${COMPS[$ci]}; D=${DESIGNS[$di]}
    echo "cell N=$N C=$C design=$D"
    $PYTHON experiments/exp_27_shape_convergence.py --only trace \
        --set "sizes=($N,)" --set "components=($C,)" --set "designs=('$D',)" \
        --set n_seeds=4 --set n_updates=800 \
        --output-dir "$RUN_ROOT/shape/N${N}_C${C}_${D}"
    ;;

*)
    echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

# Success marker, written only after the payload returned 0. `merge` steps skip
# any directory without one, so a task killed by the walltime cannot contribute
# a truncated CSV to a merged table.
MARKER="$RUN_ROOT/.ok_${MODE}_${TASK}"
mkdir -p "$(dirname "$MARKER")"
date -Is > "$MARKER"
echo "DONE mode=$MODE task=$TASK"
