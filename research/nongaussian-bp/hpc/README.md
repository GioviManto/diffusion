# Running the EM experiments on a cluster

The Layer-5 experiments are CPU-bound, pure numpy, and embarrassingly parallel
at the level of *parts*: each part of `exp_06` / `exp_07` reads nothing the
others write and emits its own CSV/PNG into the shared output directory. So a
job array with one task per part needs no merge step — the outputs simply
appear next to each other.

This directory contains SLURM array scripts. **They are templates**: the lines
marked `# SITE` need your cluster's account, partition, and module names, since
those cannot be guessed. If your cluster runs PBS/Torque or LSF instead, the
body of each script (the `srun`/`python` line) transfers unchanged — only the
directive block and the array-index variable differ.

## Quick orientation

```bash
python experiments/exp_06_em_parameter_recovery.py --list-parts
python experiments/exp_07_em_vs_score_network.py  --list-parts
```

Run one part:

```bash
python experiments/exp_06_em_parameter_recovery.py --only rate
```

Scale a part up (any key printed in `params_*.json` can be overridden):

```bash
python experiments/exp_06_em_parameter_recovery.py --only rate \
    --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,4096,16384)'
```

Unknown keys are rejected rather than ignored, so a typo in a job script fails
immediately instead of quietly running the default configuration and looking
like a finished experiment.

## Threading

Every part is dominated by `(M, M) @ (M, B)` matmuls, so BLAS threading does
most of the work within a task. Give each array task several cores and set the
thread count to match — the scripts do this from `SLURM_CPUS_PER_TASK`. Do not
leave `OMP_NUM_THREADS` unset: the default is one thread per core on the whole
node, and concurrent array tasks will then oversubscribe and slow each other
down.

## What is worth the extra compute

The single-node runs committed in `outputs/` were sized to finish in about an
hour each. The parts that would genuinely benefit from a cluster, in order:

1. **`exp_07 sample_efficiency`** — the headline curve has one replicate per
   `N`, which is why the EM curve wobbles. Repeating it over seeds would turn
   it into a curve with error bars, which is what a paper needs. Extend `sizes`
   upward too: the network was still improving at `N = 2048`, and the honest
   version of "EM-BP needs ≥64× less data" requires following the network until
   it either catches up or visibly plateaus.
2. **`exp_06 rate`** — the `N^{-1/2}` claim rests on 4 replicates. It should
   have 32+, and a decade more range in `N`.
3. **`exp_06 price_of_noising`** — 10 replicates make the sd/CRLB ratios noisy
   to about ±25%. More replicates would let the efficiency claim be stated
   sharply rather than as "within ±30%".
4. **`exp_06 quantization`** — the `M = 1601` grid is the expensive one, and
   the lattice effect would be better characterized on a denser sweep of
   `rho_true` values that are *not* low-denominator rationals.
5. **`exp_07 capacity`** — larger architectures and longer training, to push the
   "the network is not merely undertrained" argument as far as it will go.

## Files

    slurm_exp06.sbatch   5-task array, one per part of exp_06
    slurm_exp07.sbatch   4-task array, one per part of exp_07
    slurm_replicates.sbatch
                         seed-replicated sample-efficiency sweep: one task per
                         seed, each writing to its own subdirectory, plus a
                         merge helper for producing error bars
