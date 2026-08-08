#!/bin/bash
#SBATCH --job-name=wavelet
#SBATCH --partition=medium_cpu
#SBATCH --time=07:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_23/24/25 -- the wavelet-tree extension to real images.
#
# Why this needs the cluster, and why the settings are what they are.
#
# One BP pass costs ~30 ms per image-orientation at grid M = 241 and 72 ms at M = 401,
# measured rather than estimated, and it scales as M^2. One EM iteration is
# N x 3 orientations x T noise levels passes, so the fit alone is
#
#     N=2000, T=3, M=301, 20 iters  ->  12.2 min/iter  ->  4.1 h   (per family)
#     N=4000, T=3, M=401, 30 iters  ->  43.2 min/iter  -> 21.6 h   (per family)
#
# The second of those is what this script asked for in its first version and it would
# have hit the 7:45 wall and been killed, wasting a QOS slot. The settings below are
# budgeted against the measured cost with margin: ~4.1 h fit + ~0.7 h reverse sampling
# + generation, against a 7:45 limit. The numbers assume ONE core; the BLAS threads
# requested above help the GEMMs but are not counted on.
#
# If you raise n_train or grid_size, re-run the arithmetic first. Cost is linear in
# n_train and n_iters and quadratic in grid_size.
#
# Note on log output. Every python invocation below uses `-u`. Without it CPython
# block-buffers stdout when it is a file rather than a tty, so a multi-hour job writes
# nothing to the .out file until it finishes -- which makes `tail -f` on a running job
# useless and makes a hung job indistinguishable from a quiet one. The existing sbatch
# scripts in this directory do not do this; they predate the observation.
#
#   Gate first (tests plus every experiment at --quick, ~15 min):
#     SMOKE=$(sbatch --parsable --time=00:45:00 --export=ALL,MODE=smoke bocconi_wavelet.sbatch)
#
#   The gating measurement -- subband kurtosis and cross-scale dependence on 10k images:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=stats bocconi_wavelet.sbatch
#
#   The generation comparison, one array task per kernel family:
#     sbatch --dependency=afterok:$SMOKE --array=0-2 --export=ALL,MODE=generate bocconi_wavelet.sbatch
#
# The array splits by *family* rather than by part because the three parts share the
# fitted models (exp_25 caches them per process) and refitting per part would triple
# the cost. Each task writes to its own --output-dir, so there is no merge step.

set -euo pipefail

ROOT=/home/3164542/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-stats}

THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

# The CIFAR archive is gitignored, so it will not have arrived with the code.
ARCHIVE=$ROOT/research/nongaussian-bp/data/cifar-10-python.tar.gz
if [ ! -s "$ARCHIVE" ]; then
    echo "missing $ARCHIVE" >&2
    echo "fetch it once on the login node:" >&2
    echo "  mkdir -p $(dirname "$ARCHIVE")" >&2
    echo "  curl -L -o $ARCHIVE https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz" >&2
    echo "  md5sum $ARCHIVE   # expect c58f30108f718f92721af3b95e74349a" >&2
    exit 2
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE  threads=$THREADS"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    "$PYTHON" -u -m pytest tests/ -q
    for e in exp_23_wavelet_statistics exp_24_wavelet_fit exp_25_wavelet_generation; do
        echo "--- $e --quick ---"
        "$PYTHON" -u "experiments/$e.py" --quick \
            --output-dir "$ROOT/outputs/smoke/$e"
    done
    ;;

  stats)
    # The gate: if CIFAR subband coefficients came back approximately Gaussian the
    # whole premise would not apply, so this runs first and at full size.
    "$PYTHON" -u experiments/exp_23_wavelet_statistics.py \
        --output-dir "$ROOT/outputs/exp_23_wavelet_statistics" \
        --set n_images=10000 --set n_boot=400
    ;;

  generate)
    FAMILIES=(gaussian mixture scale_mixture)
    F=${FAMILIES[$TASK]}
    OUT=$ROOT/outputs/exp_25_wavelet_generation/$F
    echo "family=$F -> $OUT"
    "$PYTHON" -u experiments/exp_25_wavelet_generation.py \
        --output-dir "$OUT" \
        --set families="('$F',)" \
        --set n_train=2000 --set n_test=5000 --set n_iters=20 \
        --set grid_size=301 --set t_train="(0.3,0.6,1.0)" \
        --set n_generate=10000 --set n_reverse=128 --set reverse_steps=150
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, stats or generate)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
