Paste everything below into the same conversation where you gave me the two
previous Bocconi committee reviews of my thesis. This is the round-three
response, and it covers the thesis *and* the paper.

---

You reviewed my thesis as a senior Bocconi professor on the examining committee.
Round one: **major revisions**. Round two: **minor mandatory revisions**, with
"scientific substance: ready", "new experiment required: no", and one primary
blocker — the false universal marginal-Gaussian claim in Chapter 4.

This is the response. Please review it in the same role and hold it to the same
standard. Submission is 16 September, so this is close to the last pass.

**Read the current PDFs from GitHub before reviewing:**

- Thesis (195 pp): <https://github.com/GioviManto/diffusion/blob/main/overleaf/thesis/main.pdf>
- Paper (9 body pp + appendix): <https://github.com/GioviManto/diffusion/blob/main/overleaf/paper/main.pdf>

Every chapter, section, table and figure number below matches these exact
revisions; I checked each against the compiled table of contents rather than
against my memory of the source files. If you cannot fetch the PDFs, say so
immediately instead of reviewing from memory of the previous draft — a lot has
moved.

## The one thing I most want from you this round

Between round two and now I ran a full audit of my own — every typed number
against the frozen data that is supposed to produce it, every file path against
the repository, every cross-reference against the compiled document. It found
three things you did not flag, and one of them was worse than anything in your
list. I want to know (a) whether you agree with how I have corrected them, and
(b) what that class of error suggests you should look at that neither of us has
looked at yet.

**1. A number wrong by a factor of ten, in three documents.** Chapters 8 and 9
and the paper all said the fitted innovation shape needs "on the order of 2000
iterations" to settle. Nothing ever measured that. It came from a diagnostic
that ran one fit *out to* 2000 iterations to show its per-edge gain had
plateaued — a statement about where the gain flattens, not about where the shape
stops moving — and the two got conflated. The actual sweep says the shape
settles at a **median of 229 updates, largest observed 638**, against 80 for the
correlation coefficient. Everything downstream survives, because a 40-iteration
budget is still far short of 229, so the generative-fidelity withdrawal stands.
But the derived claims did not: "a factor of about fifty" between the two
coordinates is **2.8** — formed within each fit and then aggregated, which is
not the quotient of the two medians, 2.9 — and "93% of 72 configurations" is
**105 of 112**, counted over both run-length shards of the sweep. §8.5 now
states the population, the tolerance, and what "settled" means: the first update
after which a coordinate stays within a fixed relative tolerance of its
end-of-run value, which is trace stability rather than distance to an optimiser.
The measurement is a generator, `make_convergence_numbers.py`, and the prose
cites macros. §8.5 and §9.3.

**2. Chapter 9 was printing a table my own audit had withdrawn.** §9.5's
non-Markov table was the hand-typed one whose numbers my compendium says "could
not be reproduced from any committed output under any aggregation", and whose
baseline trained on two fifths of the frozen protocol's optimisation steps.
Undertraining the baseline inflates every ratio — in the one direction that
flatters me. The frozen-budget rerun existed and only the compendium was using
it. §9.5 is rebuilt on it and three prose numbers moved: the worst-case rank-one
ratio is 1.06 and not 2.08, the estimator's own error at the strongest coupling
is 58%, and the crossover sits at γ = 0.10.

**3. The paper and the thesis disagreed on the same quantity.** Validation-
versus-test agreement was 82.2%/79.5% typed in the paper and 81.1%/77.6%
generated in the thesis. Both now cite the macros.

I would rather you tell me these corrections are still not right than let them
through. And if you think the right response to finding this class of error is
to distrust the remaining typed numbers, say so and tell me which.

## Your round-two items

**Primary blocker — Chapter 4's marginal claim.** Fixed as you specified. The
universal claim is gone. §4.1 now carries a proposition scoped to the stationary
Gaussian chain — every noisy frame marginally N(0,1), per-frame score exactly
−x_k, zero Fisher information about α — followed by a remark giving your
characteristic-function argument: from φ_x(u) = φ_a(e^{−t}u)·exp[−(1−e^{−2t})u²/2],
exact Gaussianity of a noised frame at finite t forces the clean frame Gaussian,
so a non-Gaussian frame stays non-Gaussian under any finite corruption and its
one-frame score stays nonlinear. The general statement is now about *dependence*
rather than information. Figure 4.1's caption no longer asserts the per-frame
score is −x_k on the same page as a trimodal prior, and Chapter 7's opening no
longer claims Chapter 4 argued marginal blindness "on general grounds".

**Grid normalisation (your §5).** No bug. `MixtureInnovationKernel.log_transition_matrix`
returns the raw density on the grid with no per-column renormalisation — your
Convention 1 — so Q(θ|θ⁽ᵏ⁾) = ⟨Ξ(θ⁽ᵏ⁾), log K_θ⟩ is correct with no missing
θ-dependent log-normaliser. §8.4 now says which of the two conventions is meant,
that BP is therefore exact for the chain-shaped *factor model* rather than for a
normalised finite-state kernel, and quotes the column-mass residual as the
measured distance between them: 9.6 × 10⁻⁴ inside |a| ≤ A/2 and 1.9 × 10⁻² at the
truncated edge. I recomputed both from the kernel independently of the stored
sweep; they match.

**§9.2, the structure-aware baseline.** The number survives. Its causal reading
is gone: the chapter no longer treats the residual as an estimation term, and
gives your counter-example explicitly — an approximation floor b² makes
(b² + c_w/N)/(c_e/N) grow linearly in N for a wholly architectural reason, and
the raw risks are consistent with exactly that (EM–BP's risk falls about
4.5× across the range, the window head's by less than 2×). Two further
corrections you did not ask for: the screen's winning **radius was wrong in three
documents** — the selection rule the code implements picks radius 4, not the
radius 2 that had been read off the wrong aggregation — and the "192 cells" is
now explained as 16 seeds × 4 sizes × 3 regions and labelled a descriptive
diagnostic rather than a replication count.

**§9.3, capacity.** You were right that six seeds with no predeclared region
could not establish equivalence. The rerun landed: 16 paired seeds, two sizes,
fits run to plateau. It answers the question **in the opposite direction to the
earlier design**. No mixture capacity improves on a single Gaussian innovation at
either size; C = 1 wins on 13 of 16 seeds at N = 128 and 14 of 16 at N = 512. The
predeclared region also refuses "saturates by eight": at N = 128 the C = 8 → C = 16
step is resolved *against* the region with C = 16 worse. Two confounds travel with
it and both disfavour the larger mixtures — every cell at C ≥ 4 stopped at the
iteration cap, and the share of fits below the grid's resolution floor climbs
from 0/32 to 11/32. The old six-seed table is withdrawn, and the discrepancy is
recorded rather than quietly dropped.

**Length.** 195 pages. Appendices D–F are untouched; I would rather you tell me
whether they should go than cut them on my own judgement.

**Register.** The development-history passages you listed are gone from the body.
What remains is in Appendix C, where the history is the point: which artifact
produced which number, and what its provenance is and is not.

## What is deliberately still open

- **Generative fidelity.** Whether pointwise accuracy and generative fidelity can
  dissociate is unresolved and stays unresolved. The comparison available scores
  converged estimators against unconverged ones on the one coordinate that had
  not converged. §9.4 says that and claims nothing.
- **No two-sided locality theorem.** You said this was not a request for another
  proof, so §5.7 still reports the windowed rate as measured over r = 2…13 and
  the exact influence-coefficient decay separately.
- **One figure has no surviving plotting script.** The restyling step that
  produced the thesis's vector PDFs lived in a directory removed during a
  restructuring and was never committed. Appendix C now says so rather than
  naming a script that is not in the repository — which is what it did before.
  I would like your view on whether disclosing that is better than removing the
  figure.
- **One experiment is running as I send this.** The Laplace variant of the
  misspecification study exists only at the old, undertrained-baseline budget, so
  I dropped the claim from §9.5 rather than restate it — the defect flatters
  exactly the claim it supports. The frozen-budget rerun is in the queue.

## What I would like back

1. **A verdict on submittability**, thesis and paper separately: ready, ready
   with listed non-blocking edits, or not ready.
2. **Anything still wrong**, ranked, with blocking and non-blocking separated. If
   a number looks wrong to you, say which and why — I can check any of them
   against the frozen outputs within minutes.
3. **The audit question above**: given the three errors my own pass caught, where
   would you look next?
4. **A defence-preparation list**: the questions you would ask me in the viva
   that this document does not already answer.

Be as demanding as you were in round one. Where I have closed a point by scoping
a claim rather than by running the experiment you would have preferred, say so
plainly — I would rather hear it now than in October.
