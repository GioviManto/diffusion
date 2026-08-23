#!/usr/bin/env python
"""The structured-baseline table: EM-BP against a weight-shared 1-D CNN.

WHY THIS EXISTS. The headline table (tools/make_tab_efficiency.py) compares
EM-BP against a fully connected MLP and reports 7-20x. The MLP carries no
locality prior and no weight sharing, so the single most obvious objection to
that number is that it measures the architecture's mismatch to sequence data
rather than the value of knowing the Markov structure. Until this table existed
the paper had no answer, and worse, its prose claimed the convolutional arm was
already in the headline table when exp_07 trains no such arm at all.

exp_12 supplies the answer. A weight-shared window predictor is exactly a 1-D
CNN, its radius is swept to r = 16 -- the FULL receptive field at n_sites = 33,
so the head can see every site -- and both the radius and the eps/x0
parameterisation are chosen on a validation bundle disjoint from test. That is
the strongest baseline in this repository, and it is the one a reviewer asks
for. The gap it leaves is the part attributable to structure rather than to
architecture.

Aggregation follows the headline table exactly, because the two numbers are
read side by side: average the noise levels WITHIN a seed first, then across
seeds, and report the standard error across seeds. The twelve (here five) noise
levels share a training set and a fitted model, so they are not independent and
the seed is the inferential unit.

    python tools/make_tab_structured.py

Writes overleaf/shared/sections/tab-structured.tex and
overleaf/shared/sections/structured-numbers.tex. Do not hand-edit either.
"""
import csv
import glob
import sys

import numpy as np

# Prefer the scaled sixteen-seed sweep; fall back to the original four-seed run
# so the table exists before the sweep lands. Which one was used is printed and
# recorded in the caption, because a four-seed number must not be read as if it
# carried the frozen protocol's authority.
SCALED = "outputs/exp_12_scaled/seed*/efficiency_val.csv"
PILOT = "outputs/exp_12_receptive_field/efficiency_val.csv"
MIN_SCALED_SEEDS = 8


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


rows = load(SCALED)
source, scaled = SCALED, True
if len({r["seed"] for r in rows}) < MIN_SCALED_SEEDS:
    pilot = load(PILOT)
    if not pilot:
        print(f"REFUSING: no exp_12 output at {SCALED} or {PILOT}",
              file=sys.stderr)
        sys.exit(1)
    print(f"  scaled sweep has {len({r['seed'] for r in rows})} seeds "
          f"(< {MIN_SCALED_SEEDS}); falling back to the pilot", file=sys.stderr)
    rows, source, scaled = pilot, PILOT, False

seeds = sorted({r["seed"] for r in rows}, key=int)
sizes = sorted({int(r["n_chains"]) for r in rows})
F = lambda r, k: float(r[k])


def per_seed(g, key):
    """Mean and s.e. with the seed as the inferential unit."""
    v = np.array([np.mean([F(r, key) for r in g if r["seed"] == s])
                  for s in seeds if any(r["seed"] == s for r in g)])
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


lines, cnn_ratios = [], []
for n in sizes:
    g = [r for r in rows if int(r["n_chains"]) == n]
    em, ese = per_seed(g, "em_bp_score_rel_l2")
    cnn, cse = per_seed(g, "cnn_score_rel_l2_selected")
    mlp, mse = per_seed(g, "mlp_score_rel_l2_selected")
    # Ratios are formed per seed and then averaged, never as a ratio of means:
    # E[X]/E[Y] is not E[X/Y] and the paired design is what gives the s.e. its
    # meaning.
    rc, rcse = per_seed(g, "ratio_cnn_selected")
    rm = np.mean([F(r, "mlp_score_rel_l2_selected") / F(r, "em_bp_score_rel_l2")
                  for r in g])
    cnn_ratios.append(rc)
    lines.append(f"{n:<4} & ${mlp:.4f}$ & ${cnn:.4f} \\pm {cse:.4f}$ & "
                 f"$\\mathbf{{{em:.4f} \\pm {ese:.4f}}}$ & "
                 f"${rm:.1f}$ & ${rc:.2f} \\pm {rcse:.2f}$ \\\\")

n_cells = len(rows)
em_wins = sum(1 for r in rows
              if F(r, "em_bp_score_rel_l2") < F(r, "cnn_score_rel_l2_selected"))
agree = 100 * np.mean([int(r["cnn_selection_agrees"]) for r in rows])
radii = sorted({int(r["cnn_radius_selected"]) for r in rows})
max_radius = max(int(r["cnn_radius_oracle"]) for r in rows)
lo, hi = min(cnn_ratios), max(cnn_ratios)
n_levels = len({r["t"] for r in rows})

# Where EM-BP loses. Reported rather than left implicit: "wins 313 of 320" is
# only informative alongside where the other 7 are, and they are not scattered.
losses = [r for r in rows
          if F(r, "em_bp_score_rel_l2") >= F(r, "cnn_score_rel_l2_selected")]
loss_ns = sorted({int(r["n_chains"]) for r in losses})
loss_sizes = ", ".join(str(n) for n in loss_ns) if loss_ns else "none"
t_lo = min(float(r["t"]) for r in rows)
loss_lowt = sum(1 for r in losses if float(r["t"]) == t_lo)

provenance = ("the sixteen-seed sweep" if scaled else
              "a four-seed pilot; the sixteen-seed sweep is not yet merged")
caveat = "" if scaled else (
    " These are pilot numbers at four seeds and are not yet on the frozen "
    "protocol.")

# "the full width" is a claim about the sweep, not a phrase to keep in the
# caption whatever the data says. At n_sites = 33 a window of 2r+1 covers every
# site once r >= 16; the pilot stopped at 12 and the sentence would have
# asserted full coverage for a head that cannot see the ends of the chain.
N_SITES = 33
full_rf = max_radius >= (N_SITES - 1) // 2
rf_phrase = (
    f"whose receptive-field radius is swept to $r={max_radius}$, the full "
    f"width at $\\nsites={N_SITES}$, so the largest head sees every site"
    if full_rf else
    f"swept to $r={max_radius}$ (a window of ${2 * max_radius + 1}$ of "
    f"$\\nsites={N_SITES}$ sites; the full-width sweep is pending)"
)

tex = f"""%% GENERATED by tools/make_tab_structured.py from {source}
%% ({len(seeds)} seeds, {len(sizes)} sizes, {n_levels} noise levels, {n_cells} cells).
%% Do not hand-edit the numbers; rerun the generator.

%% An inline centred tabular rather than a float, matching the house style the
%% Hellinger table already uses for a secondary result: the body has a nine-page
%% limit and a float's caption block plus its placement slack cost most of a
%% page here. The explanation lives in the surrounding prose instead.
\\begin{{center}}\\small
\\captionof{{table}}{{The CNN is a weight-shared window predictor, {rf_phrase};
radius and parameterisation chosen on a validation bundle disjoint from test.
{len(seeds)} seeds $\\pm$ one s.e., per seed.{caveat}}}
\\label{{tab:structured}}
\\begin{{tabular}}{{rccccc}}
\\toprule
$\\nseq$ & MLP & CNN (tuned) & EM--BP & MLP/EM & CNN/EM \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""

macros = f"""%% GENERATED by tools/make_tab_structured.py -- do not hand-edit.
%%
%% The prose cites these and never types the numbers, for the reason given in
%% tools/make_tab_efficiency.py: a typed number and a computed number drift, and
%% nothing catches it.
\\newcommand{{\\cnnratiolo}}{{{lo:.1f}}}
\\newcommand{{\\cnnratiohi}}{{{hi:.1f}}}
\\newcommand{{\\cnnseeds}}{{{len(seeds)}}}
\\newcommand{{\\cnncells}}{{{n_cells}}}
\\newcommand{{\\cnnemwins}}{{{em_wins}}}
\\newcommand{{\\cnnagree}}{{{agree:.0f}}}
\\newcommand{{\\cnnmaxradius}}{{{max_radius}}}
\\newcommand{{\\cnnscaled}}{{{'true' if scaled else 'false'}}}
%% Where the CNN wins, so the body can say it rather than imply it never does.
\\newcommand{{\\cnnlosses}}{{{n_cells - em_wins}}}
\\newcommand{{\\cnnlosssizes}}{{{loss_sizes}}}
\\newcommand{{\\cnnlossloweest}}{{{loss_lowt}}}
"""

for path, blob, label in (
    ("../../overleaf/shared/sections/tab-structured.tex", tex, "table"),
    ("../../overleaf/shared/sections/structured-numbers.tex", macros, "macros"),
):
    with open(path, "w") as fh:
        fh.write(blob)
    print(f"wrote {path}")

print(f"  source: {provenance}")
print(f"  CNN/EM-BP ratio {lo:.2f}-{hi:.2f} across nseq={sizes}")
print(f"  EM-BP beats the tuned CNN in {em_wins}/{n_cells} cells; "
      f"radii selected {radii}")
