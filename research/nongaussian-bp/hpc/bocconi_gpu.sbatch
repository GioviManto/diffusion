#!/bin/bash
#SBATCH --job-name=bpgpu
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# GPU runs. The BP recursion is the same code as on CPU, parameterised by its array module;
# see src/backend.py for why there is deliberately no second implementation.
#
# THE VERSION PIN IS LOAD-BEARING. cupy-cuda12x must stay below 14. CuPy 14.x JIT-compiles
# its cub segmented-reduction kernels against headers this cluster's toolkit cannot build
# (42 errors inside cuda_fp4.hpp), and the failure is partial and misleading: cuBLAS matmuls
# still succeed, so a GPU looks usable right up until the first .max(axis=...) in the
# recursion. cupy 13.6.0 compiles every operation the recursion needs. Verified on an H200:
# tests/test_backend_parity.py 12/12, device agreement 1.8e-16 to 9.3e-15, GPU against the
# closed-form Gaussian score 4.2e-16.
#
#   Families sweep:  sbatch --array=0-4 --export=ALL,MODE=families bocconi_gpu.sbatch
#   Grid refinement: sbatch --array=0-2 --export=ALL,MODE=grid     bocconi_gpu.sbatch
#   Chain length:    sbatch --array=0-2 --export=ALL,MODE=length   bocconi_gpu.sbatch

set -euo pipefail

source /home/3164542/nongaussian-bp/gpu_env.sh
export BP_DEVICE=gpu

ROOT=/home/3164542/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-families}

echo "host=$(hostname) task=$TASK mode=$MODE"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" -c "from src.backend import get_xp, device_name; print('device:', device_name(get_xp()))"
echo "start=$(date -Is)"

# Refuse to run the sweep if the device path is not exact. A silent numpy fallback or a
# broken kernel would otherwise produce plausible numbers on the wrong hardware.
"$PYTHON" -m pytest tests/test_backend_parity.py -q || { echo "PARITY FAILED - refusing to run"; exit 3; }
echo

case "$MODE" in
  families)
    # Does the exp_16 finding -- that the pointwise ranking does not transfer to the
    # generated law -- hold beyond the Laplace chain? Five innovation families at matched
    # covariance, so they differ ONLY beyond second moments.
    FAMS=(laplace student uniform mixture gaussian)
    F=${FAMS[$TASK]}
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate,pointwise \
        --output-dir "$ROOT/outputs/exp_16/family_${F}" \
        --set family="'$F'" --set sizes="(128,512,2048)" --set n_seed=3 --set n_steps=400
    ;;
  grid)
    # Grid convergence OF THE GENERATION RESULT. Every distributional number so far is at
    # M=401; if the ranking moved with M it would be quadrature, not score quality.
    MS=(401 801 1601)
    M=${MS[$TASK]}
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate \
        --output-dir "$ROOT/outputs/exp_16/grid_M${M}" \
        --set grid_m="$M" --set sizes="(2048,)" --set n_seed=3 --set n_steps=400
    ;;
  length)
    NS=(32 64 128)
    N=${NS[$TASK]}
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate \
        --output-dir "$ROOT/outputs/exp_16/len_n${N}" \
        --set n_sites="$N" --set sizes="(2048,)" --set n_seed=3 --set n_steps=400
    ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
echo "end=$(date -Is)"
