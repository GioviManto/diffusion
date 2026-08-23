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
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
          20: "twenty", 21: "twenty-one", 22: "twenty-two"}

# The certified run (job 630845): same protocol and seeds as exp_07_symmetric,
# plus the resolution certificate, the density-level Hellinger columns, and the
# corrected ECM inner budget. exp_07_symmetric_seed* is its predecessor and is
# kept only so the two can be compared; it must not be the source, because it
# carries no em_resolved and the gate below would then have nothing to check.
SOURCE = "outputs/frozen/exp_07_certified_seed*/sample_efficiency_val.csv"

# Sizes beyond the certified grid, each run at a raised budget because the
# certified caps were chosen for nseq <= 2048. That budget difference is not
# assumed harmless -- it is measured, by CALIB below.
#
#   nseq=4096  jobs 631496/631497, H200, em_iters=2400, net_steps=60000
#   nseq=8192  job 633361, H200, em_iters=3200, net_steps=100000
#
# 8192 only became affordable once the EM E-step went on the device: until then
# BP_DEVICE reached bp_grid and denoiser but not src/em.py, so the arm that
# dominates the cost ran on the CPU whatever partition the job was sent to.
EXTENDED = (
    "outputs/frozen/exp_07_n4096_seed*/sample_efficiency_val.csv",
    "outputs/frozen/exp_07_n8192_seed*/sample_efficiency_val.csv",
)

# nseq=2048 at the raised budget (job 631467), the same seeds and bundles as the
# certified run. This exists to answer one question and no other: the caption used
# to call the largest ratio "budget-limited" because both arms often selected
# their largest allowed budget there. Selecting the cap is not the same as being
# bounded by it, and the difference is measurable -- so it was measured, by
# rerunning one size with both caps raised and pairing on seed.
CALIB = "outputs/frozen/exp_07_budget2048_seed*/sample_efficiency_val.csv"
CALIB_N = 2048


def load(pattern, what):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"REFUSING: no {what} run found at {pattern}", file=sys.stderr)
        sys.exit(1)
    out = []
    for f in files:
        seed = f.split("seed")[1].split("/")[0]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["seed"] = seed
                out.append(r)
    return out


rows = load(SOURCE, "certified")
ext_blocks = [load(p, f"extended-nseq ({p.split('exp_07_')[1].split('_seed')[0]})")
              for p in EXTENDED]
calib = load(CALIB, "budget-calibration")

seeds = sorted({r["seed"] for r in rows}, key=int)
F = lambda r, k: float(r[k])

EXPECTED = 16
checks = [("certified", rows), ("calibration", calib)]
checks += [(f"extended[{i}]", b) for i, b in enumerate(ext_blocks)]
for label, block in checks:
    s = sorted({r["seed"] for r in block}, key=int)
    if len(s) != EXPECTED:
        print(f"REFUSING: {label} has {len(s)} seeds, expected {EXPECTED}: "
              f"{','.join(s)}", file=sys.stderr)
        sys.exit(1)
    # The calibration is a PAIRED comparison and the extended row is aggregated
    # per seed alongside the rest; both are meaningless if the seeds differ.
    if s != seeds:
        print(f"REFUSING: {label} seeds differ from certified", file=sys.stderr)
        sys.exit(1)

# Each extended run must contribute exactly one size, and one the table does not
# already have. Two runs landing on the same nseq would silently average two
# different budgets into one row, which is the failure this whole file exists to
# prevent.
base_sizes = {int(r["n_chains"]) for r in rows}
seen = set(base_sizes)
ext_ns = []
for pattern, block in zip(EXTENDED, ext_blocks):
    s = {int(r["n_chains"]) for r in block}
    if len(s) != 1 or (s & seen):
        print(f"REFUSING: {pattern} covers {sorted(s)}, which is not a single "
              f"new size beyond {sorted(seen)}", file=sys.stderr)
        sys.exit(1)
    n = s.pop()
    seen.add(n)
    ext_ns.append(n)
    rows = rows + block

# The largest extended size is the one the refusal message names; the caption
# names them all, since a reader checking which rows share a protocol should not
# have to infer it from the largest.
EXT_N = max(ext_ns)
sizes = sorted(seen)
_ns = sorted(ext_ns)
# Named from the globs actually loaded, so the provenance header cannot go on
# crediting exp_07_n4096_seed*/ for rows that came from somewhere else.
ext_provenance = " and ".join(
    p.split("/")[2] + "/" for p in EXTENDED
)

# The verb has to agree with the number of extended rows, or the caption reads
# "The nseq = 4096 and nseq = 8192 rows uses a raised budget" the moment a
# second size is added -- which is exactly what happened when 8192 landed.
ext_phrase = (f"The $\\nseq = {_ns[0]}$ row uses"
              if len(_ns) == 1 else
              "The $\\nseq = " + "$ and $\\nseq = ".join(str(n) for n in _ns)
              + "$ rows use")

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
lo, hi = min(ratios), max(ratios)

# The budget calibration.
#
# The previous caption asserted that the ratio at the largest size was
# "budget-limited", inferring it from how often each arm selected its largest
# allowed budget. That inference is wrong, and the error is instructive: an arm
# selects the last checkpoint whenever validation error is still falling, however
# slowly, so cap-selection measures the SIGN of the remaining gain and says
# nothing about its SIZE.
#
# So the size was measured. exp_07_budget2048 reruns nseq=2048 with the EM cap
# tripled and the network's cap tripled, on the same seeds and the same
# validation and test bundles, which makes the comparison paired -- necessary
# here, because the seed-to-seed spread in the ratio is an order of magnitude
# larger than the effect being looked for.
def paired_delta(a_rows, b_rows, n, key="ratio_selected"):
    """Per-seed mean of `key` at size `n`, differenced seed by seed."""
    def by_seed(block):
        return np.array([np.mean([F(r, key) for r in block
                                  if int(r["n_chains"]) == n and r["seed"] == s
                                  and ("em_resolved" not in r or int(r["em_resolved"]))])
                         for s in seeds])
    d = by_seed(b_rows) - by_seed(a_rows)
    return d.mean(), d.std(ddof=1) / np.sqrt(d.size)


cal_d, cal_se = paired_delta(rows, calib, CALIB_N)
cal_net = paired_delta(rows, calib, CALIB_N, "net_score_rel_l2_selected")[0]
cal_net_base = np.mean([F(r, "net_score_rel_l2_selected") for r in rows
                        if int(r["n_chains"]) == CALIB_N])

# If tripling both budgets moved the ratio by more than this, the caps really were
# binding and the caption must say so rather than dismiss it. The measured value
# is -0.16 +/- 0.18, so the branch below reports "does not move"; the threshold is
# here so that a future rerun which DOES move cannot quietly keep the old wording.
CALIB_MAX = 1.0
if abs(cal_d) > CALIB_MAX:
    print(f"REFUSING: tripling both budgets moves the nseq={CALIB_N} ratio by "
          f"{cal_d:+.2f} (> {CALIB_MAX}). The caps bind, so the table must not "
          f"carry an nseq={EXT_N} row run at a different budget, and the caption "
          f"must state the limitation rather than dismiss it.", file=sys.stderr)
    sys.exit(1)

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
%% exp_07_certified_seed*/ (nseq <= {CALIB_N}) and {ext_provenance}
%% ({len(seeds)} seeds, {n_cells} cells). The budget calibration in the caption comes
%% from exp_07_budget2048_seed*/. Do not hand-edit the numbers; rerun the generator.

\\begin{{table}}[!htbp]
\\caption{{Relative denoising error against grid BP under the true kernel, averaged over the noise
schedule; {len(seeds)} seeds $\\pm$ one standard error, aggregated per seed. Both arms are tuned on a
disjoint validation bundle --- the network selects its parameterisation \\emph{{and}} training
length, EM--BP its iteration count --- agreeing with the test-set optimum in ${net_agree:.1f}\\%$ and
${em_agree:.1f}\\%$ of cells. EM--BP is more accurate in ${th(n_cells - wins)}$ of ${th(n_cells)}$.
{ext_phrase} a raised budget, the caps having been set for $\\nseq \\le {CALIB_N}$;
rerunning $\\nseq = {CALIB_N}$ at that budget on the same seeds moves its ratio by
${cal_d:+.2f} \\pm {cal_se:.2f}$.{drop_note}}}
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

# NOTHING IS WRITTEN UNTIL BOTH FILES ARE BUILT.
#
# The table used to be written here, before the macro block below. That made the
# two outputs non-atomic, and the failure is not hypothetical: when the ratio
# first exceeded eighteen, the deliberate KeyError in _WORDS fired *after* the
# table had already been written, leaving a nine-row table quoting 20.2 beside a
# macro file still saying \ratiohi{17.6} and \biggestn{4096}. The paper would
# have built, with the prose and the table disagreeing -- which is the single
# thing this generator exists to make impossible.
#
# So both strings are constructed first and written only once neither can fail.

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
%%
%% \\ratiolosub, \\ratiohisub, \\ratiohisubword and \\nsizessub USED TO BE DEFINED
%% HERE and deliberately are not any more. They existed to quote the range over
%% the sizes at which neither arm selected its largest budget, because the largest
%% size was believed to be budget-limited. That belief was tested by rerunning
%% nseq={CALIB_N} with both budgets tripled and found to be false. Deleting the
%% macros rather than redefining them is the point: any prose still carrying the
%% old framing fails to build instead of silently rendering a number that no
%% longer means what the surrounding sentence says it means.
\\newcommand{{\\ratiolo}}{{{lo:.1f}}}
\\newcommand{{\\ratiohi}}{{{hi:.1f}}}
\\newcommand{{\\ratioloword}}{{{_WORDS[round(lo)]}}}
\\newcommand{{\\ratiohiword}}{{{_WORDS[round(hi)]}}}
\\newcommand{{\\nfreeparams}}{{{N_FREE}}}
\\newcommand{{\\nseedsused}}{{{len(seeds)}}}
\\newcommand{{\\ncellsused}}{{{th(n_cells)}}}
\\newcommand{{\\nsizesused}}{{{len(sizes)}}}
\\newcommand{{\\biggestn}}{{{big}}}
\\newcommand{{\\netagree}}{{{net_agree:.1f}}}
\\newcommand{{\\emagree}}{{{em_agree:.1f}}}
%% The budget calibration, quoted wherever the text claims the comparison is not
%% an artefact of the optimisation caps.
\\newcommand{{\\nseqcalib}}{{{CALIB_N}}}
\\newcommand{{\\budgetdelta}}{{{cal_d:+.2f}}}
\\newcommand{{\\budgetdeltase}}{{{cal_se:.2f}}}
"""
mdest = "../../overleaf/shared/sections/efficiency-numbers.tex"

# Both strings exist and neither construction raised. Now write.
open(dest, "w").write(tex)
open(mdest, "w").write(macros)

print(f"wrote {dest}: {len(seeds)} seeds, {n_cells} cells, "
      f"{len(sizes)} sizes (to nseq={big}), ratio {lo:.1f}-{hi:.1f}")
print(f"wrote {mdest}: \\ratiolo={lo:.1f} \\ratiohi={hi:.1f} "
      f"\\biggestn={big} \\nfreeparams={N_FREE}")
print(f"  network val==test {net_agree:.1f}%, EM {em_agree:.1f}%, "
      f"network wins {wins} cells")
print(f"  budget calibration at nseq={CALIB_N}: ratio {cal_d:+.2f} +/- {cal_se:.2f}, "
      f"network error {100*cal_net/cal_net_base:+.1f}% (does not bind)")
