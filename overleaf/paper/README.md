# paper — the full paper

**Recovering Markov dynamics from noised sequences: exact scores and transition
learning by local inference.** Nine pages of main content, which is the NeurIPS
limit, plus unlimited references and appendix.

Compile `main.tex` from inside this folder. `appendix.tex` is inputted by it and
by nothing else.

## The claim, in one sentence

A locally specified non-Gaussian Markov law induces a globally dependent
diffusion score, yet that score is computable by exact inference on the latent
chain — and the same inference produces the pairwise posterior statistics needed
to learn the local transition.

## Structure

One estimation question asked at three levels of difficulty, each adding exactly
one unknown to the one below: the kernel is given, then two numbers are unknown,
then the whole transition density is. Then one section on what supplying the
Markov structure is worth against trained denoisers.

## What is deliberately not here

- **The rotating ring.** Different state space, different estimand, different
  likelihood. It is a separate result and lives in `../compendium/` (ch12) and
  `../thesis/` (ch09).
- **Reverse generation, the capacity sweep, non-Markov stress tests, the
  efficient-information table.** Exploratory or unresolved; the compendium keeps
  them with their status attached.
- **Correction history.** The paper presents accepted claims. What was withdrawn
  and why is the compendium's job.

## The one number that needs care

The efficiency ratio is real but asymmetric by construction: the estimator is
handed the Markov factorisation, the autoregressive form and homogeneity, and
the networks are handed none of them. It measures what that structure is worth
on this family. It is not a ranking of learning algorithms, and the paper says so
in the abstract, at the table, and in the limitations.

At the largest sample size both arms select the largest budget on offer, so that
row's ratio is bounded by the grid rather than by the data. The caption says
which row and quotes the unbounded range alongside.
