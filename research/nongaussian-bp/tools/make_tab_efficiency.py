"""Emit the efficiency table and its macros into overleaf/shared/sections/.

Generated rather than hand-typed, because this table is the paper's headline and
it has already been wrong twice: once from an EM budget nobody had checked, and
once from selecting EM's budget while pinning the network's.

Aggregation is PER SEED and then across seeds. Cells within a seed share one
training set and one set of fitted models, so the twelve noise levels move
together; treating 84 cells as 84 independent observations would misstate the
interval in both directions (it understates it for the ratio and overstates it
for the error columns).
"""
import csv, glob, sys
import numpy as np

sys.path.insert(0, "experiments")
from frozen_config import FROZEN  # noqa: E402

# Derived, never typed. theta is [rho, pi (C), mu (C), s2 (C)] = 1 + 3C numbers,
# of which pi is simplex-constrained, so 3C are free. The table said "12 free"
# -- correct at C=4, and the frozen config has used C=8 since the paired-design
# sweep, so the headline column was understating the estimator by half.
N_FREE = 3 * FROZEN.n_components

# The abstract spells small integers rather than setting them in maths. Only the
# range the prose actually quotes needs covering; a KeyError here is the right
# failure, because it means the ratio moved somewhere the wording was not written
# for and the sentence needs rereading, not a wider lookup table.
_WORDS = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen"}

# The certified run (job 630845): same protocol and seeds as exp_07_symmetric,
# plus the resolution certificate, the density-level Hellinger columns, and the
# corrected ECM inner budget. exp_07_symmetric_seed* is its predecessor and is
# kept only so the two can be compared; it must not be the source, because it
# carries no em_resolved and the gate below would then have nothing to check.
SOURCE = "outputs/frozen/exp_07_certified_seed*/sample_efficiency_val.csv"
FILES = sorted(glob.glob(SOURCE))
if not FILES:
    print(f"REFUSING: no certified run found at {SOURCE}", file=sys.stderr)
    sys.exit(1)
rows = []
for f in FILES:
    seed = f.split("seed")[1].split("/")[0]
    with open(f) as fh:
        for r in csv.DictReader(fh):
            r["seed"] = seed
            rows.append(r)

seeds = sorted({r["seed"] for r in rows}, key=int)
sizes = sorted({int(r["n_chains"]) for r in rows})
F = lambda r, k: float(r[k])

EXPECTED = 16
if len(seeds) != EXPECTED:
    print(f"REFUSING: {len(seeds)} seeds present, expected {EXPECTED}: "
          f"{','.join(seeds)}", file=sys.stderr)
    sys.exit(1)

# Resolution gate.
#
# An under-resolved mixture component -- narrower than two grid cells -- can raise
# the quadrature likelihood without corresponding to anything in the innovation
# law, so a cell fitted that way is not evidence. Those cells are dropped.
#
# Dropping rather than refusing outright, because the question that matters is
# not "were any cells unresolved" but "does the answer depend on them". So the
# exclusion is measured: if removing the unresolved cells moves any ratio by more
# than MAX_SHIFT, the result rested on fits that are lattice artefacts and must
# not ship. If it does not, the result is robust and the drop is bookkeeping.
#
# Measured on the certified run: 16 of 1344 cells, 13 of them at the smallest
# sample size and 11 from a single seed, almost all having run to the iteration
# cap -- small sample, long budget, one component collapsing onto the mesh. The
# largest ratio shift from excluding them is 0.081.
UNRESOLVED_FRAC_MAX = 0.05
MAX_SHIFT = 0.5

dropped = []
if "em_resolved" in (rows[0] if rows else {}):
    dropped = [r for r in rows if not int(r["em_resolved"])]
    frac = len(dropped) / len(rows)
    if frac > UNRESOLVED_FRAC_MAX:
        print(f"REFUSING: {len(dropped)} of {len(rows)} cells ({100*frac:.1f}%) "
              f"have a mixture component under two grid cells wide. That is too "
              f"many to treat as bookkeeping -- refit on a finer grid or with a "
              f"stronger variance floor.", file=sys.stderr)
        sys.exit(1)
else:
    print("  resolution gate: NOT CERTIFIED -- these outputs predate "
          "em_resolved. Rerun exp_07 part5 to certify.", file=sys.stderr)

all_rows = rows
rows = [r for r in rows if "em_resolved" not in r or int(r["em_resolved"])]


def per_seed(g, key):
    v = np.array([np.mean([F(r, key) for r in g if r["seed"] == s]) for s in seeds])
    return v.mean(), v.std(ddof=1) / np.sqrt(v.size)


lines, ratios = [], []
for n in sizes:
    g = [r for r in rows if int(r["n_chains"]) == n]
    nm, nse = per_seed(g, "net_score_rel_l2_selected")
    em, ese = per_seed(g, "em_bp_score_rel_l2")
    rm, _ = per_seed(g, "ratio_selected")
    ratios.append(rm)
    lines.append(f"{n:<4} & ${nm:.4f} \\pm {nse:.4f}$ & "
                 f"$\\mathbf{{{em:.4f} \\pm {ese:.4f}}}$ & ${rm:.1f}$ \\\\")

n_cells = len(rows)
net_agree = 100 * np.mean([int(r["net_steps_agrees"]) for r in rows])
em_agree = 100 * np.mean([int(r["em_iters_agrees"]) for r in rows])
wins = sum(1 for r in rows if F(r, "ratio_selected") < 1)
big = sizes[-1]
gb = [r for r in rows if int(r["n_chains"]) == big]
net_cap = 100 * np.mean([int(r["net_steps_selected"]) == 20000 for r in gb])
em_cap = 100 * np.mean([int(r["em_iters_selected"]) == 400 for r in gb])
lo, hi = min(ratios), max(ratios)
sub = [r for i, r in enumerate(ratios) if sizes[i] != big]

# Does the answer depend on the cells that were dropped?
#
# This is the check that makes dropping them legitimate rather than convenient.
# Recompute every ratio on the FULL set and compare: if excluding the
# under-resolved fits moves a ratio materially, the published number rested on
# lattice artefacts and must not ship, however few of them there were.
if dropped:
    shifts = []
    for i, n in enumerate(sizes):
        g_all = [r for r in all_rows if int(r["n_chains"]) == n]
        v = np.array([np.mean([F(r, "ratio_selected") for r in g_all if r["seed"] == s])
                      for s in seeds])
        shifts.append(abs(ratios[i] - v.mean()))
    worst_i = int(np.argmax(shifts))
    if shifts[worst_i] > MAX_SHIFT:
        print(f"REFUSING: excluding the {len(dropped)} under-resolved cell(s) "
              f"moves the ratio at n={sizes[worst_i]} by {shifts[worst_i]:.3f} "
              f"(> {MAX_SHIFT}). The result depends on fits whose narrowest "
              f"mixture component is thinner than the mesh; refit before "
              f"reporting.", file=sys.stderr)
        sys.exit(1)
    print(f"  resolution gate: dropped {len(dropped)} of {len(all_rows)} cells "
          f"(worst s_min/h {min(float(r['em_s_min_over_h']) for r in dropped):.2f}); "
          f"largest ratio shift {max(shifts):.3f}, at n={sizes[worst_i]}")
else:
    print(f"  resolution gate: all {len(rows)} cells resolved "
          f"(min s_min/h {min(float(r['em_s_min_over_h']) for r in rows):.2f})")

# Density-level recovery, reported alongside the score error it can outlive.
hell = np.array([F(r, "em_hellinger") for r in rows]) if "em_hellinger" in rows[0] else None

# Stated in the caption rather than left to a footnote: a reader is entitled to
# know that cells were excluded, how many, and that excluding them did not move
# the answer.
drop_note = ""
if dropped:
    drop_note = (f" {len(dropped)} of {len(all_rows)} cells are excluded because the "
                 f"narrowest fitted mixture component is under two grid cells wide; "
                 f"including them changes no ratio by more than ${max(shifts):.2f}$.")

body = "\n".join(lines)
# `$1,344$` sets the comma as a binary-operator-ish relation with the wrong
# spacing; `$1{,}344$` is the correct way to write a thousands separator in math
# mode, and is what the hand-written table used.
th = lambda x: f"{x:,}".replace(",", "{,}")
tex = f"""%% Included by overleaf/paper/main.tex only. The workshop no longer carries this table:
%% at four pages it presents the analytical result, and a comparison whose whole
%% content is a protocol cannot be defended in the space available.
%%
%% GENERATED by tools/make_tab_efficiency.py from outputs/frozen/
%% exp_07_symmetric_seed*/ ({len(seeds)} seeds, {n_cells} cells). Do not hand-edit
%% the numbers; rerun the generator.

\\begin{{table}}[!htbp]
\\caption{{Relative denoising error against grid BP under the true kernel, averaged over the noise
schedule; {len(seeds)} seeds $\\pm$ one standard error, aggregated per seed. Both arms are tuned on a
disjoint validation bundle --- the network selects its parameterisation \\emph{{and}} training
length, EM--BP its iteration count --- agreeing with the test-set optimum in ${net_agree:.1f}\\%$ and
${em_agree:.1f}\\%$ of cells. EM--BP is more accurate in ${th(n_cells - wins)}$ of ${th(n_cells)}$.
At $\\nseq = {big}$ both select their largest allowed budget in ${net_cap:.0f}\\%$ and
${em_cap:.0f}\\%$ of cells, so that ratio is budget-limited; elsewhere the range is
${min(sub):.1f}$--${max(sub):.1f}$.{drop_note}}}
\\label{{tab:pointwise}}
\\centering
\\begin{{tabular}}{{rccc}}
\\toprule
sequences $\\nseq$ & network & EM--BP (${N_FREE}$ free) & ratio \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

dest = "../../overleaf/shared/sections/tab-efficiency.tex"
open(dest, "w").write(tex)

# The prose gets macros, not typed numbers.
#
# The abstract said "between 8 and 14" for three weeks after this generator
# started emitting 7.3-15.7, and no check caught it because one number was typed
# in main.tex and the other computed here. A gate comparing the two would need a
# tolerance, and the tolerance would be tuned until it passed -- so instead there
# is nothing to compare: the prose cites \ratiolo and cannot hold a stale value.
macros = f"""%% GENERATED by tools/make_tab_efficiency.py -- do not hand-edit.
%%
%% Every efficiency number the prose quotes is defined here and nowhere else, so
%% that the abstract, the body and the table cannot disagree. If a value looks
%% wrong, rerun the generator; do not edit the number in the text.
\\newcommand{{\\ratiolo}}{{{lo:.1f}}}
\\newcommand{{\\ratiohi}}{{{hi:.1f}}}
\\newcommand{{\\ratiolosub}}{{{min(sub):.1f}}}
\\newcommand{{\\ratiohisub}}{{{max(sub):.1f}}}
\\newcommand{{\\ratioloword}}{{{_WORDS[round(lo)]}}}
\\newcommand{{\\ratiohisubword}}{{{_WORDS[round(max(sub))]}}}
\\newcommand{{\\nfreeparams}}{{{N_FREE}}}
\\newcommand{{\\nseedsused}}{{{len(seeds)}}}
\\newcommand{{\\ncellsused}}{{{th(n_cells)}}}
\\newcommand{{\\nsizesused}}{{{len(sizes)}}}
\\newcommand{{\\nsizessub}}{{{len(sub)}}}
\\newcommand{{\\biggestn}}{{{big}}}
\\newcommand{{\\netagree}}{{{net_agree:.1f}}}
\\newcommand{{\\emagree}}{{{em_agree:.1f}}}
"""
mdest = "../../overleaf/shared/sections/efficiency-numbers.tex"
open(mdest, "w").write(macros)

print(f"wrote {dest}: {len(seeds)} seeds, {n_cells} cells, "
      f"ratio {lo:.1f}-{hi:.1f} (excl. n={big}: {min(sub):.1f}-{max(sub):.1f})")
print(f"wrote {mdest}: \\ratiolo={lo:.1f} \\ratiohi={hi:.1f} "
      f"\\ratiolosub={min(sub):.1f} \\ratiohisub={max(sub):.1f} "
      f"\\nfreeparams={N_FREE}")
print(f"  network val==test {net_agree:.1f}%, EM {em_agree:.1f}%, "
      f"network wins {wins} cells")
