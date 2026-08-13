#!/bin/bash
#SBATCH --job-name=bpem_validate
#SBATCH --partition=short_cpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/3164542/bp-em-diffusion/logs/%x_%j.out
#SBATCH --error=/home/3164542/bp-em-diffusion/logs/%x_%j.err
#
# Run the full validation suite on the cluster before trusting any sweep output.
# The sweeps are submitted with --dependency=afterok on this job, so if the foundation
# is broken on this machine nothing downstream burns compute on bad numbers.

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

PYTHON=/home/3164542/bp-em-diffusion/.venv/bin/python
cd /home/3164542/bp-em-diffusion/code

echo "host=$(hostname)  start=$(date -Is)"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo

echo "########## test_bp_core.py ##########"
"$PYTHON" test_bp_core.py

echo
echo "########## test_em_parametric.py ##########"
"$PYTHON" test_em_parametric.py

echo
echo "end=$(date -Is)  ALL VALIDATION PASSED"
