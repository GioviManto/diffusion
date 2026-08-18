# workshop — the analytical front half

**Exact diffusion scores for non-Gaussian Markov chains: what second-order
message closure computes.** Four pages of main content; currently 3, so there is
a page of headroom.

Compile `main.tex` from inside this folder.

## This is not an abridgement of the paper

It is a smaller, different claim. The paper asks what can be recovered from
noised sequences and answers it at three levels. This note makes one point:

> The score of a non-Gaussian Markov chain is computable by exact local
> inference, and the standard second-order approximation to that inference is
> exactly a named estimator — the LMMSE estimator of the covariance-matched
> Gaussian — rather than an uncontrolled one.

One proposition, one figure, no table.

## What is deliberately not here

- **The efficiency comparison and its table.** A comparison whose entire content
  is a protocol cannot be defended in four pages, and a reader who cannot check
  the protocol should not be shown the number.
- **The rotating ring and the blindness theorem.** Separate result; see the
  compendium and thesis.
- The discretisation-control experiment, the gradient-ascent-versus-EM
  comparison, the convergence-rate asymmetry, the mixture extension.

Learning appears as one section that states Fisher's identity in prose and points
at the paper for the derivations and experiments.

## If you spend the spare page

Put it into the functional-BP exposition: what a message *is* as a function on
the reals, and why the grid is a representation of it rather than the thing
itself. That is the part readers of the earlier draft found compressed.
