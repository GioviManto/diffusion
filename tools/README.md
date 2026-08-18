# tools — repository-level scripts

Two one-off analyses that span the whole repository rather than any single
experiment. Everything that belongs to the research package lives in
`../research/nongaussian-bp/tools/` instead — that is where the gates, the
figure and table generators, and the reproducibility checks are.

| Script | What it does |
|---|---|
| `audit_em_bp_provenance.py` | traces a reported EM/BP number back to the run and configuration that produced it |
| `summarize_generation_rerun.py` | summarises a reverse-generation rerun against the previous one |

## Where the scripts you probably want actually are

```
research/nongaussian-bp/tools/
  check_paper.sh          the gate: page limits, placeholders, typed numbers
  check_all.sh            everything, including the test suite
  check_replicates.py     every experiment's replicate count against the frozen config
  check_reproducible.py   re-runs and diffs deterministic outputs
  make_tab_efficiency.py  writes the efficiency table and its macros into overleaf/shared
  make_figures.py         writes every figure into overleaf/shared/figures
  merge_replicates.py     merges sharded cluster output
  pull_and_check.sh       rsync results down from the cluster and validate them
```

`make_handover.sh` used to live here. It assembled a flat copy of the four
documents, and was deleted on 18 August 2026 when `overleaf/` became the single
canonical home — there are no copies left to keep in sync.
