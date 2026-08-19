#!/bin/bash
#SBATCH --job-name=frozen
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/3164542/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=/home/3164542/nongaussian-bp/logs/%x_%A_%a.err
#
# Do NOT inherit the submitting shell's resource limits. The login node runs with
# `ulimit -t 600`, and Slurm propagates the submitter's rlimits into the job by
# default -- so a job with --time=06:00:00 is still killed by SIGXCPU after ten
# minutes of CPU time, which at --cpus-per-task=4 is about seven minutes of wall
# clock. Array 630828 lost all six running shards to exactly this, each dying at
# TotalCPU=10:00.4 with "CPU time limit exceeded (core dumped)" while the log
# showed nothing wrong.
#
# It bites harder the more threads are asked for, and it did not bite the earlier
# waves only because they finished inside the limit.
#SBATCH --propagate=NONE
#
# THE FROZEN BATCH. Every number in the paper comes from this script.
#
# The problem it exists to solve: the previous draft stitched results together
# from runs at 40, 120 and 200 EM iterations, at 3, 6 and 16 replicates, and at
# rho = 0.8 and 0.85. Numbers from those runs could not go in the same table,
# several conclusions turned out to be artefacts of the iteration budget, and
# the paper ended up carrying its own corrections. Every experiment below now
# imports experiments/frozen_config.py; none sets rho, the grid, or an
# iteration count locally, and tools/check_paper.sh fails if one starts to.
#
# ---------------------------------------------------------------------------
# SUBMISSION. Pin the stamp ONCE, before the gate, and pass it to every wave.
#
#   S=$(date +%Y%m%d)
#   GATE=$(sbatch --parsable --partition=short_cpu --time=01:00:00 \
#          --export=ALL,RUN_STAMP=$S,MODE=smoke hpc/bocconi_frozen.sbatch)
#   sbatch --dependency=afterok:$GATE --array=0-7 \
#          --export=ALL,RUN_STAMP=$S hpc/bocconi_frozen.sbatch
#   sbatch --dependency=afterok:$GATE --array=8-17%6 --partition=compute \
#          --time=1-00:00:00 --export=ALL,RUN_STAMP=$S hpc/bocconi_frozen.sbatch
#
# 19 tasks including the gate, against a QOS ceiling of 30 submitted and 10
# running. The %6 on wave B leaves slots for wave A, which is shorter and
# feeds more of the paper.
#
# ---------------------------------------------------------------------------
# WHY CPU, HAVING CHECKED THE GPU OPTION RATHER THAN ASSUMED IT.
#
# The recursion has a working CuPy backend: src/backend.py is one implementation
# parameterised by its array module, fp64 throughout, and
# tests/test_backend_parity.py measures device agreement at 1.8e-16 to 9.3e-15
# on an H200. The E-step is a batched GEMM, which is exactly what a GPU is for.
# So the question is real, and the answer is a scheduling one, not a numerical
# one. Measured on 2026-08-16, before this batch was sized:
#
#   medium_gpuh200   13 PENDING          gpuh200    15 PENDING
#   gpua100          15 PENDING          medium_gpua100  9 PENDING
#   medium_cpu       cnode08 IDLE, five more in mix-
#
# Three things follow. (1) The GPU queues are the congested ones here; a task
# sent there would sit behind thirteen. (2) The binding constraint on this sweep
# is the QOS ceiling of 10 RUNNING tasks, not per-task speed, so ten parallel
# CPU tasks beat two faster GPU ones. (3) The networks in E9 have 25,248
# parameters -- a GPU is not the bottleneck for a model that size.
#
# So this batch runs on CPU, and the GPU path stays validated rather than
# retired: MODE=smoke runs the parity suite, and MODE=gpu below is a working
# escape hatch for when the GPU queues clear or the grid grows enough that
# O(d N_g^2) per sequence stops fitting in a CPU walltime. Set
# --partition=medium_gpuh200 --gres=gpu:1 and MODE=gpu. If cupy imports but
# parity fails, the run ABORTS -- it does not fall back to CPU and quietly
# relabel itself, because a silent fallback is how a GPU result becomes a CPU
# result with a GPU provenance stamp. Keep cupy-cuda12x BELOW 14: 14.x
# JIT-compiles its cub reductions against headers this toolkit cannot build, and
# the failure is partial -- matmuls succeed, .max(axis=) fails -- so a GPU looks
# usable right up to the first reduction in the recursion.

set -euo pipefail

ROOT=/home/3164542/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"
mkdir -p "$ROOT/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-array}
STAMP=${RUN_STAMP:-$(date +%Y%m%d)}
RUN_ROOT="outputs/frozen"
mkdir -p "$RUN_ROOT"

echo "=== frozen batch ==="
echo "task=$TASK mode=$MODE stamp=$STAMP host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1}"
$PYTHON -c "
import sys; sys.path.insert(0,'experiments')
from frozen_config import FROZEN, provenance
print('rho=%s  d=%s  N_g=%s  A=%s  seeds=%d  t-levels=%d'
      % (FROZEN.rho, FROZEN.n_sites, FROZEN.n_grid, FROZEN.half_width,
         FROZEN.n_seeds, len(FROZEN.t_grid)))
assert provenance()['is_frozen'], 'frozen_config has been edited away from FROZEN'
"

# ---------------------------------------------------------------------------
# MODE=smoke -- the gate. Nothing else may start until this returns 0.
#
# A sweep launched against a stale or half-synced checkout burns days and
# produces numbers indistinguishable from the ones being replaced. This runs the
# whole suite (including the ring normaliser tests, which pin a bug whose
# failure mode is a smooth confident wrong answer) and a --quick pass of every
# experiment that feeds the paper.
# ---------------------------------------------------------------------------
if [ "$MODE" = "smoke" ]; then
    echo "--- pytest ---"
    $PYTHON -m pytest tests/ -q
    echo "--- quick pass of every experiment in the batch ---"
    $PYTHON experiments/exp_28_ring_em.py --quick --only rotation
    $PYTHON experiments/exp_08_gradient_vs_exact_mstep.py --quick
    $PYTHON experiments/exp_06_em_parameter_recovery.py --quick --only rate
    echo "GATE PASSED"
    exit 0
fi

if [ "$MODE" = "gpu" ]; then
    source "$ROOT/gpu_env.sh"
    export BP_DEVICE=gpu
    # Abort rather than fall back. See the header.
    $PYTHON -m pytest tests/test_backend_parity.py -q
fi

# ---------------------------------------------------------------------------
# The task table.
#
# Sharded by PART, not by seed. `common.experiment_parser` documents parts as
# independent and writing disjoint CSVs precisely so a scheduler can run them as
# parallel array tasks into one --output-dir with no merge step -- so this
# sharding cannot produce a half-written table, which seed-sharding into a
# shared file can.
#
# Wave A (0-7)   medium_cpu, 6h    -- Rungs 0, 1, 2 and the ring
# Wave B (8-17)  compute, 24h      -- Rung 3 and the network comparison
# ---------------------------------------------------------------------------
run() { echo "+ $*"; "$@"; }
OUT="--output-dir $RUN_ROOT"

case "$TASK" in
  # --- Wave A -------------------------------------------------------------
  0)  # E1 + E2: Rung 0. Discretisation control and the LMMSE gap.
      run $PYTHON experiments/exp_01_grid_validation.py $OUT/exp_01
      run $PYTHON experiments/exp_02_laplace_gaussian_message_error.py $OUT/exp_02
      ;;
  1)  # E2 cont: the gap across five variance-matched innovation families.
      run $PYTHON experiments/exp_03_nongaussian_innovation_sweep.py $OUT/exp_03
      ;;
  2)  # E3: Rung 1. Gradient ascent against the exact M-step.
      run $PYTHON experiments/exp_08_gradient_vs_exact_mstep.py $OUT/exp_08
      ;;
  3)  # E3 cont: (rho, q) recovery and the n^{-1/2} rate.
      run $PYTHON experiments/exp_06_em_parameter_recovery.py $OUT/exp_06 \
          --only rate,monotonicity
      ;;
  4)  # E4: Rung 2. Mixture recovery diagnostics and the fitted density.
      run $PYTHON experiments/exp_18_revision_diagnostics.py --out $RUN_ROOT/exp_18 \
          --parts emtrace,density
      ;;
  30|31|32|33)
      # E6: Rung 4a, the confining potential. Sharded four ways by seed.
      #
      # Task 5 ran this unsharded and was killed by the six-hour wall having
      # written nothing: write_csv is called once, after the last cell. The
      # cost is 16 seeds x 5 sizes x 6 noise levels x 2 arms = 960 gradient
      # fits, and the arms are wildly asymmetric -- the joint arm plateaus in
      # 40-80 steps while the marginal arm runs to the 300 cap every time, at
      # ~0.2 s/step either way. That is ~10 h against a 6 h partition limit.
      #
      # Seeds are independent (part_potential reseeds per seed), so a shard is
      # exactly a subset of the same run, not an approximation of it.
      SHARD=$((TASK - 30))
      SEEDS=$($PYTHON -c "
import sys; sys.path.insert(0,'experiments')
from frozen_config import FROZEN
print(tuple(list(FROZEN.seeds())[$SHARD::4]))")
      run $PYTHON experiments/exp_28_ring_em.py $OUT/exp_28_ring_em_p$SHARD \
          --only potential --set "seeds=$SEEDS"
      ;;
  8[0-7])
      # Rung 4a, SECOND attempt. The first (628942) completed all four shards
      # and produced nothing usable in the joint arm: part_potential generated
      # two species (+-OMEGA) while fit_potential scores every trajectory at one
      # scalar psi, so half the data was evaluated under the wrong rotation. The
      # estimator escaped by collapsing the ring -- r_star pinned to the lower
      # clip in 52% of cells, because a radius-zero ring is rotation-invariant
      # and so pays nothing for the wrong psi. Fixed by matching part_rotation
      # and using one species; tests/test_ring.py pins both directions.
      #
      # Also raised from 300 steps to 600: in 628942 the blind arm hit the cap
      # in 90% of cells and the joint arm in 28%, so the previous budget was
      # measuring the cap and not the estimator. Eight shards of two seeds keeps
      # that inside the six-hour wall.
      SHARD=$((TASK - 80))
      SEEDS=$($PYTHON -c "
import sys; sys.path.insert(0,'experiments')
from frozen_config import FROZEN
print(tuple(list(FROZEN.seeds())[$SHARD::8]))")
      run $PYTHON experiments/exp_28_ring_em.py $OUT/exp_28_ring_em_v2_p$SHARD \
          --only potential --set "seeds=$SEEDS" --set potential_steps=600
      ;;
  9[0-9]|10[0-5])
      # Rung 4a, THIRD attempt. Submit to `compute`, NOT medium_cpu -- see the
      # wall-clock note at the bottom of this block.
      #
      # v2 (629193) fixed the species defect: clip pinning fell from 51% to 5%
      # and the joint arm went back to beating the marginal one. Its arm
      # comparison is still not reportable, because the two arms were not held
      # to the same convergence standard -- at t<=1 the marginal arm hit the
      # 600-step cap in 95% of cells against the joint arm's 57%. "Joint beats
      # marginal" then partly means "joint converged and marginal did not", and
      # the obvious control does not rescue it: only 13 of 400 pairs had both
      # arms converged.
      #
      # WHAT THIS RUN CAN AND CANNOT SETTLE. A probe refitting twelve cells to
      # 2400 steps with the plateau test disabled was first read off its single
      # first cell (n=512, t=0.2216), where the joint arm is frozen from step 600
      # and the marginal arm improves 0.482 -> 0.372 -> 0.335. Across all twelve
      # that does not hold: the joint arm moves in five of six cells and at
      # n=512, t=0.4665 gets WORSE with more steps (0.029 at 600, 0.060 at 2400),
      # and the parameter error is non-monotone in four cells while the
      # log-likelihood rises monotonically in all twelve.
      #
      # So more optimisation is not more accuracy -- the finite-sample maximiser
      # is not at the truth, and the budget is a regularisation knob, the same
      # thing exp_29 found for EM. This run holds both arms to ONE n-independent
      # criterion, which removes a defect that was certainly there. It does not
      # produce converged estimators, and the margin it reports must be read with
      # that caveat. Selecting each arm's stopping point on held-out data, as
      # exp_07 now does for both of its arms, is what would actually settle it.
      #
      # The root cause is the stopping rule, not the budget. plateau_tol was an
      # ABSOLUTE tolerance on an objective that sums over sequences, so it was
      # eight times stricter at n=4096 than at n=512 -- a convergence criterion
      # varying along the very axis this rung reports its scaling on. It is now
      # per-sequence (exp_28_ring_em.fit_potential), pinned by
      # tests/test_ring.py::test_stopping_rule_does_not_depend_on_sample_size.
      #
      # Budget, from the cap_probe timings rather than from optimism. At n=512
      # a 2400-step joint fit is 240s and a marginal fit 562s, and cost is
      # linear in n. A whole seed at cap 3000 -- 5 sizes, 6 noise levels, both
      # arms -- is about 19 hours, because the joint arm parks early and stops
      # but the marginal arm runs to the cap. Sixteen of those is not a sensible
      # ask, and medium_cpu's 6:10 wall already killed 628434_5.
      #
      # So: eight seeds, split by SIZE into two shards each. Sizes are grouped
      # {128,512,1024,2048} against {4096}, which is balanced -- 7.25 against
      # 8.0 in units of n/512 -- because cost is linear in n and 4096 alone is
      # half the total work. Each shard is then ~10h, which `compute` (3-day
      # limit) takes comfortably.
      #
      # Eight seeds rather than sixteen: v2 already has sixteen for the
      # n-scaling, and the arm comparison this run exists to settle has median
      # error ratios of 1.35x and 1.72x, which eight seeds over 5 sizes and 6
      # noise levels (240 matched pairs) resolves easily.
      SHARD=$((TASK - 90))
      SEED_IDX=$((SHARD / 2))
      if [ $((SHARD % 2)) -eq 0 ]; then SIZES="(128, 512, 1024, 2048)"
      else                              SIZES="(4096,)"; fi
      SEEDS=$($PYTHON -c "
import sys; sys.path.insert(0,'experiments')
from frozen_config import FROZEN
print((list(FROZEN.seeds())[$SEED_IDX],))")
      run $PYTHON experiments/exp_28_ring_em.py $OUT/exp_28_ring_em_v3_p$SHARD \
          --only potential --set "seeds=$SEEDS" \
          --set "potential_sizes=$SIZES" --set potential_steps=3000
      ;;
  6)  # E7 + E8: Rungs 4b and 4c, plus the density panel.
      run $PYTHON experiments/exp_28_ring_em.py $OUT/exp_28_ring_em \
          --only rotation,recovery,density
      ;;
  7)  # Compendium: the efficient-information sweep. Not a paper claim -- the
      # exponents are noisy Monte Carlo Fisher estimates -- but the ORDERING
      # (shape decays fastest) is robust and worth recording.
      run $PYTHON experiments/exp_22_shape_information.py $OUT/exp_22
      ;;

  # --- Wave B -------------------------------------------------------------
  8|9|10|11)
      # E5: Rung 3. Convergence rates, rho against innovation shape. Sharded
      # four ways over the (size, component) cells; each writes its own CSV.
      SHARD=$((TASK - 8))
      case $SHARD in
        0) S="(128,)"  ; C="(4,)"  ;;
        1) S="(512,)"  ; C="(4,)"  ;;
        2) S="(128,)"  ; C="(8,)"  ;;
        3) S="(512,)"  ; C="(8,)"  ;;
      esac
      run $PYTHON experiments/exp_27_shape_convergence.py \
          $OUT/exp_27_shard$SHARD --only trace \
          --set "sizes=$S" --set "components=$C" --set n_updates=400
      ;;
  12|13|14|15|18|19|20|21|22|23|24|25|26|27|28|29)
      # E9: THE HEADLINE. Sample efficiency against the neural baselines.
      #
      # Sharded by seed, one seed per task, sixteen in total to match
      # FROZEN.n_seeds. Tasks 12-15 carry seeds 0-3 and 18-29 carry seeds 4-15;
      # the gap is because 16 and 17 were already spoken for when the second
      # group was added.
      #
      # Worth stating, because it is the kind of gap this whole configuration
      # exists to prevent: exp_07 takes a single `seed` rather than a replicate
      # count, so `provenance()['is_frozen']` stays true for a run with four
      # replicates instead of sixteen. The frozen config cannot police a knob
      # the experiment does not expose. Sixteen tasks here is what actually
      # honours it.
      if [ "$TASK" -le 15 ]; then SEED=$((TASK - 12)); else SEED=$((TASK - 14)); fi
      run $PYTHON experiments/exp_07_em_vs_score_network.py \
          $OUT/exp_07_seed$SEED \
          --only sample_efficiency,sample_efficiency_val \
          --set "seed=$SEED"
      ;;
  4[0-9]|5[0-5])
      # E9 RERUN at the frozen iteration budget.
      #
      # The first sixteen-seed run above was sound in every respect except one:
      # exp_07 hard-coded `n_iters=120` in four places, while
      # FROZEN.em_max_iters said 400 and nothing in the tree read it. Same
      # disease as the replicate knobs of tools/check_replicates.py -- a frozen
      # setting the experiment never asks for -- and `is_frozen` was true
      # throughout.
      #
      # It is the conservative direction: an under-iterated EM makes the
      # estimator look WORSE, so the published 7.9-13.7x is a lower bound, not
      # an inflated one. Rerunning anyway, because a headline number should be
      # the one the protocol specifies.
      SEED=$((TASK - 40))
      run $PYTHON experiments/exp_07_em_vs_score_network.py \
          $OUT/exp_07_i400_seed$SEED \
          --only sample_efficiency,sample_efficiency_val \
          --set "seed=$SEED"
      ;;
  6[0-9]|7[0-5])
      # E9 FINAL: both arms tuned by the same protocol.
      #
      # The 400-iteration rerun above answered the wrong question. Holding the EM
      # budget fixed at ANY value is a choice that favours one arm: measured on
      # array 628943, 120 beats 400 below nseq~256 and loses above it, because
      # the budget is a bias/variance knob and truncating EM is implicit
      # regularisation. Meanwhile the network has always had its parameterisation
      # chosen on a disjoint validation bundle.
      #
      # So EM-BP now gets its iteration count chosen the same way, on the same
      # bundle, per noise level -- symmetric with the network, and pre-registered
      # here rather than picked after seeing test error. Every row also carries
      # `*_at_cap`, so the cost of selecting is measurable rather than asserted.
      SEED=$((TASK - 60))
      run $PYTHON experiments/exp_07_em_vs_score_network.py \
          $OUT/exp_07_valsel_seed$SEED \
          --only sample_efficiency_val \
          --set "seed=$SEED"
      ;;
  11[0-9]|12[0-5])
      # E9 FINAL, actually symmetric this time. Supersedes 629091 (tasks 60-75).
      #
      # 629091 selected EM-BP's iteration count on validation while the network
      # trained to a fixed 20,000 steps. That was the same fairness error the
      # block above was written to fix, moved rather than removed: training
      # length is a bias/variance knob for BOTH arms -- exp_29 shows EM's
      # held-out error has an interior minimum at every sample size, and SGD's
      # does too -- so selecting one budget and pinning the other favours the
      # selected arm. It favoured us, which is exactly why it could not ship.
      #
      # The network's training length is now chosen on the same bundle, jointly
      # with its parameterisation, over net_checkpoints(n_steps) =
      # {500, 1000, 2000, 3500, 6000, 10000, 14000, 20000}. Each arm now has one
      # selected quantity; if anything the network still has the better deal,
      # selecting over a 2-D grid against EM-BP's 1-D.
      #
      # Cost note: the checkpoints are free in training time -- the grid is
      # snapshotted from a single run, not eight runs -- but evaluation now
      # scores 16 network states per noise level instead of 2, so the wall time
      # is dominated by evaluation rather than by SGD. Sixteen seeds, one per
      # shard, as before.
      SEED=$((TASK - 110))
      run $PYTHON experiments/exp_07_em_vs_score_network.py \
          $OUT/exp_07_symmetric_seed$SEED \
          --only sample_efficiency_val \
          --set "seed=$SEED"
      ;;
  13[0-9]|14[0-5])
      # E9 CERTIFIED. Supersedes 629985 + 630108 (tasks 110-125).
      #
      # Same protocol, same seeds, same selection rules. Three things the earlier
      # run could not report, all added on 18 Aug 2026:
      #
      #   1. A resolution certificate per cell (em_resolved, em_s_min_over_h).
      #      Without it there is no record of whether the fitted mixture's
      #      narrowest component was wider than the mesh it lives on, and an
      #      under-resolved component raises the quadrature likelihood without
      #      corresponding to anything in the innovation law. Spot-checked at
      #      s_min/h ~ 9.4-10.0, so this is expected to pass -- but
      #      make_tab_efficiency.py currently prints "NOT CERTIFIED" for the
      #      headline table, and "the check did not run" is not "the check
      #      passed".
      #
      #   2. Density-level recovery (em_hellinger). Every other number measures
      #      the fitted kernel through the score it induces, which at moderate
      #      noise is forgiving enough that two visibly different innovation laws
      #      agree to three decimals. The transition claim has to be stated on
      #      the transition.
      #
      #   3. The corrected ECM inner budget (n_inner 4 -> 16). At 40 outer
      #      iterations the old default left the fitted excess kurtosis at 1.68
      #      against a truth of 3.0, where 16 sweeps reach 2.14. The effect
      #      vanishes by 60 outer iterations, and 30% of cells in the previous
      #      run selected fewer than that.
      #
      # RESULT (630845, all 16 shards, ~1h each). I predicted the corrected inner
      # budget would move the EM arm in its own favour. It did the opposite: the
      # ratios came down slightly at every size -- 7.3/10.1/12.6/15.7 became
      # 7.0/9.7/12.5/15.5. Ascending Q harder made the fitted kernel a little
      # worse on held-out score error, which is the same non-monotonicity already
      # seen in the outer loop and in Rung 4a.
      #
      # The resolution certificate also earned its keep immediately: 16 of 1344
      # cells came back with a mixture component under two grid cells wide, 13 of
      # them at nseq=32 and 11 from a single seed. My spot check had found
      # s_min/h of 9.4-10.0 and concluded the fits were resolved with room to
      # spare; that was two configurations and did not generalise.
      #
      # Wall time: the previous shards ran 35-43 min. The M-step is roughly 3x
      # dearer and the E-step dominates at large nseq, so budget ~2x and keep the
      # 6-hour partition wall.
      SEED=$((TASK - 130))
      run $PYTHON experiments/exp_07_em_vs_score_network.py \
          $OUT/exp_07_certified_seed$SEED \
          --only sample_efficiency_val \
          --set "seed=$SEED"
      ;;
  16) # E5 cont: what the channel costs, against the no-noising baseline.
      run $PYTHON experiments/exp_06_em_parameter_recovery.py $OUT/exp_06 \
          --only clean_vs_noised,clean_raw_mle
      ;;
  17) # E5 cont: the misspecified case -- Laplace truth, mixture family.
      run $PYTHON experiments/exp_06_em_parameter_recovery.py $OUT/exp_06 \
          --only misspecified,price_of_noising
      ;;
  *)  echo "no task $TASK"; exit 1 ;;
esac

# The marker is written only after the payload returns 0, so a walltime kill
# leaves no marker and the collection step can tell a finished cell from a
# killed one. `set -e` above means a non-zero payload never reaches this line.
touch "$RUN_ROOT/.ok_task${TASK}"
echo "TASK $TASK DONE"
