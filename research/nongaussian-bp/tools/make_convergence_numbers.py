#!/usr/bin/env python
"""How many EM iterations each coordinate of the fit actually needs.

WHY THIS EXISTS. Three documents carried the sentence "the fitted innovation
shape needs on the order of 2000 iterations to settle." Nothing measured that.
The number came from a diagnostic in exp_32 that ran a representative C=16 fit
OUT TO 2000 iterations to show its per-edge gain had plateaued and would never
reach the strict tolerance -- a statement about where the gain flattens, not
about where the shape stops moving. The two got conflated, and the conflated
version was then used to justify withdrawing the generative-fidelity claim and
to set the capacity design's budget.

The actual measurement is exp_27, and it is an order of magnitude smaller: the
shape settles at a median of ~230 updates with a tail to ~640, against ~80 for
the correlation coefficient. The conclusions all survive -- a 40-iteration
budget is still far short of 230, so the generative comparison was still
under-converged on the one coordinate it scored -- but the factor was wrong by
about ten, and a factor wrong by ten in a thesis is worth one generator.

`settle_X` in that experiment is the first update after which coordinate X stays
within a relative tolerance of its final value (rho 1e-3, innovation variance
1e-2, excess kurtosis 2e-2). Because "final" means the value at the end of the
run, the statistic is censored by run length: a coordinate still drifting at the
cap reports a settling time near the cap. The sweep has two run lengths, and only
the longer one is uncensored, so the headline numbers come from that shard and
the censored one is reported as the lower bound it is.

    python tools/make_convergence_numbers.py

Writes overleaf/shared/sections/convergence-numbers.tex. Do not hand-edit it.
"""
import csv
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import load_params, require_clean  # noqa: E402

SOURCE_GLOB = "outputs/frozen/exp_27_*"
COORDS = {
    "settle_rho": "correlation coefficient",
    "settle_innovation_var": "innovation variance",
    "settle_innovation_excess_kurtosis": "innovation shape",
}
# The fixed budget several earlier experiments used as a default, and the one
# the chapters compare against.
LEGACY_BUDGET = 40

rows = []
for f in sorted(glob.glob(f"{SOURCE_GLOB}/shape_settle.csv")):
    rows += list(csv.DictReader(open(f)))
if not rows:
    print(f"REFUSING: no settling output under {SOURCE_GLOB}", file=sys.stderr)
    sys.exit(1)

# LEGACY, and disclosed rather than waved through. This sweep ran on 16 Aug
# 2026, before hpc/deploy_clean.sh existed, so its records carry a SLURM job id
# and an empty git commit instead of a source-archive digest. The gate is right
# to refuse it by default; the alternative to allowing it here is to quote the
# same numbers by hand, which is how the wrong one survived in the first place.
# Appendix C names the sweep and says what its provenance is and is not.
require_clean(load_params(sorted(glob.glob(SOURCE_GLOB))), allow_legacy=True)

caps = sorted({int(r["n_updates"]) for r in rows})
F = lambda r, k: float(r[k])

# Censoring check, per run length. A shard in which any configuration settles
# within 5% of its own cap cannot bound the tail, only the median from below.
near_cap = {c: sum(1 for r in rows if int(r["n_updates"]) == c
                   and max(F(r, k) for k in COORDS) >= 0.95 * c)
            for c in caps}
uncensored = [c for c in caps if near_cap[c] == 0]
if not uncensored:
    print("REFUSING: every run length in this sweep has configurations settling "
          "at its cap, so no shard measures the tail rather than the cap.",
          file=sys.stderr)
    sys.exit(1)
# The longest uncensored run length: the most headroom, hence the least
# understated tail.
CAP = max(uncensored)
head = [r for r in rows if int(r["n_updates"]) == CAP]

print(f"{len(rows)} configurations over run lengths {caps}; "
      f"headline shard = {CAP} updates ({len(head)} configurations, "
      f"{len({r['seed'] for r in head})} seeds)")
for c in caps:
    n = sum(1 for r in rows if int(r["n_updates"]) == c)
    flag = "uncensored" if near_cap[c] == 0 else f"CENSORED {near_cap[c]}/{n}"
    print(f"  cap {c:>4}: {n:>3} configurations  {flag}")

# The headline shard is one cell of the design, not the whole sweep, and the
# prose has to say which: a settling time depends on the size, the capacity and
# the noise schedule it was measured at.
def only(field, conv=str):
    v = {conv(r[field]) for r in head}
    if len(v) != 1:
        print(f"REFUSING: the {CAP}-update shard spans {field} = {sorted(v)}; "
              "this generator reports it as a single configuration and the "
              "prose describes it as one.", file=sys.stderr)
        sys.exit(1)
    return v.pop()


head_size, head_comp, head_design = (only("n_chains", int),
                                     only("n_components", int),
                                     only("design"))
print(f"  headline configuration: nseq={head_size}, C={head_comp}, "
      f"{head_design} noise design")

stat = {}
for k, label in COORDS.items():
    v = sorted(F(r, k) for r in head)
    stat[k] = (st.median(v), max(v))
    print(f"  {label:<26} median {st.median(v):>5.0f}  max {max(v):>5.0f}")

# The ratio is formed per configuration and then aggregated, not as a ratio of
# the two medians: the two coordinates are measured on the same fit, so the
# pairing is real and throwing it away would overstate the spread.
ratios = sorted(F(r, "settle_innovation_excess_kurtosis")
                / max(F(r, "settle_rho"), 1.0) for r in head)
ratio_med = st.median(ratios)
print(f"  shape / coefficient, paired  median {ratio_med:.1f}  max {max(ratios):.1f}")

# How badly the legacy fixed budget misses, over EVERY configuration in the
# sweep rather than the headline shard alone: censoring can only make a
# settling time look SMALLER, so a configuration that misses the budget in a
# censored shard misses it in truth too, and the count is therefore safe.
unsettled = sum(1 for r in rows
                if F(r, "settle_innovation_excess_kurtosis") > LEGACY_BUDGET)
print(f"  not settled in shape by {LEGACY_BUDGET} iterations: "
      f"{unsettled}/{len(rows)}")

shape_med, shape_max = stat["settle_innovation_excess_kurtosis"]
rho_med, rho_max = stat["settle_rho"]

tex = f"""%% GENERATED by tools/make_convergence_numbers.py -- do not hand-edit.
%%
%% Measured on {SOURCE_GLOB}, the {CAP}-update shard, which is the only run
%% length in that sweep where no configuration settles at its own cap. The
%% shorter shard is censored and its numbers are lower bounds, so it is not used
%% here. These macros exist because the number they replace -- "on the order of
%% 2000 iterations" -- was never measured and was wrong by about a factor of ten.
\\newcommand{{\\convseeds}}{{{len({r['seed'] for r in head})}}}
\\newcommand{{\\convsize}}{{{head_size}}}
\\newcommand{{\\convcomps}}{{{head_comp}}}
\\newcommand{{\\convdesign}}{{{head_design}}}
\\newcommand{{\\convcells}}{{{len(head)}}}
\\newcommand{{\\convcap}}{{{CAP}}}
\\newcommand{{\\convshapemed}}{{{shape_med:.0f}}}
\\newcommand{{\\convshapemax}}{{{shape_max:.0f}}}
\\newcommand{{\\convrhomed}}{{{rho_med:.0f}}}
\\newcommand{{\\convivarmed}}{{{stat["settle_innovation_var"][0]:.0f}}}
\\newcommand{{\\convrhomax}}{{{rho_max:.0f}}}
\\newcommand{{\\convratio}}{{{ratio_med:.1f}}}
%% The fixed budget the earlier experiments used, and how much of the sweep it
%% leaves unsettled in the shape coordinate. Counted over every configuration,
%% censored shards included: censoring shortens a settling time, so a
%% configuration that misses the budget there misses it in truth as well.
\\newcommand{{\\convbudget}}{{{LEGACY_BUDGET}}}
\\newcommand{{\\convunsettled}}{{{unsettled}}}
\\newcommand{{\\convunsettledof}}{{{len(rows)}}}
\\newcommand{{\\convunsettledpct}}{{{100 * unsettled / len(rows):.0f}}}
"""

dest = "../../overleaf/shared/sections/convergence-numbers.tex"
with open(dest, "w") as fh:
    fh.write(tex)
print(f"\nwrote {dest}")
