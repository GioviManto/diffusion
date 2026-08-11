#!/bin/bash
#SBATCH --job-name=wavelet
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_23/25 (real images) and exp_26 (video) -- the wavelet-tree extension.
#
# TIME LIMIT. `medium_cpu` caps at 6:10:00 (sinfo), so the 07:45:00 this script
# asked for in its previous version would have been rejected at submit rather
# than run and truncated. 05:45 leaves margin under the cap. Anything longer
# belongs on `compute` (3-00:10:00, 3 nodes) -- see MODE=video below.
#
# COST, measured rather than estimated. One BP pass is ~30 ms per
# image-orientation on a uniform M=241 grid; per-depth grids resolving to
# t = 0.05 cost about the same in total while reaching 18x further down in t
# (they spend points where the likelihood is narrow, which is where the nodes
# are few). Measured on this laptop:
#
#   exp_25 fit: 600 images, 12 iters, 3 orientations, 2 noise levels -> ~9 min
#               so n_train=2000, n_iters=20 -> ~50 min per family
#   exp_26 fit: 300 sequences x 6 frames, 8 iters                    -> ~20 min
#               (the caterpillar traverses each frame's tree twice)
#   exp_26 cutcurve: ~25 min per cut depth at those settings
#
# Cost is linear in n_train and n_iters and quadratic in grid size; the grid
# sizes follow from t_resolve, so *lowering t_resolve raises cost sharply*.
# Re-run the arithmetic before raising anything.
#
# LOG BUFFERING. Every python call uses `-u`. Without it CPython block-buffers
# stdout to a file, so a multi-hour job writes nothing until it exits and a hung
# job is indistinguishable from a quiet one.
#
# USAGE
#
#   Gate first (tests + every experiment at --quick, ~20 min):
#     SMOKE=$(sbatch --parsable --partition=short_cpu --time=01:00:00 \
#             --export=ALL,MODE=smoke bocconi_wavelet.sbatch)
#
#   Images -- the gating measurement on 10k CIFAR images:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=stats bocconi_wavelet.sbatch
#
#   Images -- generation, one array task per kernel family:
#     sbatch --dependency=afterok:$SMOKE --array=0-2 --export=ALL,MODE=generate \
#            bocconi_wavelet.sbatch
#
#   Video -- the space/time trade-off, one array task per cut depth:
#     sbatch --dependency=afterok:$SMOKE --array=0-4 --export=ALL,MODE=cutcurve \
#            bocconi_wavelet.sbatch
#
#   Video -- caterpillar vs frozen control, coherence, structure, strips:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=video bocconi_wavelet.sbatch
#
# The image array splits by *family* and the video array by *cut depth*, because
# in both cases the tasks share nothing: each fits its own model and writes its
# own file, so there is no merge step. Splitting by experiment *part* instead
# would refit the same models once per part.

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
    "$PYTHON" -u -m pytest tests/ -q -m "not slow"
    for e in exp_23_wavelet_statistics exp_24_wavelet_fit \
             exp_25_wavelet_generation exp_26_video; do
        echo "--- $e --quick ---"
        "$PYTHON" -u "experiments/$e.py" --quick \
            --output-dir "$ROOT/outputs/smoke/$e"
    done
    ;;

  stats)
    # The gate: if CIFAR subband coefficients came back approximately Gaussian
    # the whole premise would not apply, so this runs first and at full size.
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
        --set t_resolve=0.05 --set t_train="(0.3,0.6,1.0)" \
        --set t_likelihood=0.05 \
        --set n_generate=10000 --set n_reverse=128 --set reverse_steps=150
    ;;

  cutcurve)
    # One cut depth per task. Each is an independent fit of a *different exact
    # model* -- the space/time edge trade of video_bp.cut_caterpillar_bp -- and
    # writes cutcurve_cut<N>.csv, so the five concatenate with no merge logic.
    OUT=$ROOT/outputs/exp_26_video
    echo "cut_depth=$TASK -> $OUT"
    "$PYTHON" -u experiments/exp_26_video.py --only cutcurve \
        --output-dir "$OUT" \
        --set cuts="($TASK,)" \
        --set n_train=1000 --set n_test=1000 --set n_iters=12 \
        --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    ;;

  video)
    # Caterpillar against the frozen-time control, plus the coherence
    # decomposition, the loop-free budget, and the frame strips.
    "$PYTHON" -u experiments/exp_26_video.py \
        --only fit,coherence,structure,strip \
        --output-dir "$ROOT/outputs/exp_26_video" \
        --set n_train=1000 --set n_test=1000 --set n_iters=12 \
        --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, stats, generate, cutcurve or video)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
