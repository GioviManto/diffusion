"""Emit paper/sections/tab-efficiency.tex from the symmetric E9 run.

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

FILES = sorted(glob.glob("outputs/frozen/exp_07_symmetric_seed*/sample_efficiency_val.csv"))
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

# Resolution gate. An under-resolved mixture component -- narrower than two grid
# cells -- can raise the quadrature likelihood without corresponding to anything
# in the innovation law, so a cell fitted that way is not evidence and must not
# reach the table. The column is absent from runs made before it was recorded;
# those are reported as uncertified rather than silently accepted, because
# "the check did not run" and "the check passed" are different states.
if "em_resolved" in (rows[0] if rows else {}):
    bad = [r for r in rows if not int(r["em_resolved"])]
    if bad:
        worst = min(float(r["em_s_min_over_h"]) for r in bad)
        print(f"REFUSING: {len(bad)} of {len(rows)} cells have a mixture "
              f"component under two grid cells wide (worst s_min/h = "
              f"{worst:.2f}). Refit on a finer grid or with a stronger variance "
              f"floor; an unresolved component may not count as evidence.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  resolution gate: all {len(rows)} cells resolved "
          f"(min s_min/h = {min(float(r['em_s_min_over_h']) for r in rows):.2f})")
else:
    print("  resolution gate: NOT CERTIFIED -- these outputs predate "
          "em_resolved. Rerun exp_07 part5 to certify.", file=sys.stderr)


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

body = "\n".join(lines)
# `$1,344$` sets the comma as a binary-operator-ish relation with the wrong
# spacing; `$1{,}344$` is the correct way to write a thousands separator in math
# mode, and is what the hand-written table used.
th = lambda x: f"{x:,}".replace(",", "{,}")
tex = f"""%% Included by paper/main.tex only. The workshop no longer carries this table:
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
${min(sub):.1f}$--${max(sub):.1f}$.}}
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

dest = "paper/sections/tab-efficiency.tex"
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
mdest = "paper/sections/efficiency-numbers.tex"
open(mdest, "w").write(macros)

print(f"wrote {dest}: {len(seeds)} seeds, {n_cells} cells, "
      f"ratio {lo:.1f}-{hi:.1f} (excl. n={big}: {min(sub):.1f}-{max(sub):.1f})")
print(f"wrote {mdest}: \\ratiolo={lo:.1f} \\ratiohi={hi:.1f} "
      f"\\ratiolosub={min(sub):.1f} \\ratiohisub={max(sub):.1f} "
      f"\\nfreeparams={N_FREE}")
print(f"  network val==test {net_agree:.1f}%, EM {em_agree:.1f}%, "
      f"network wins {wins} cells")
