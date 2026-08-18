# compendium — the development record

Every derivation the paper states, every experiment that was run including the
ones that were cut, and the audit trail of what was claimed and later withdrawn.
Internal document: it is written for us, not for a referee.

Compile `main.tex` from inside this folder. Chapters are in `chapters/`.

## What this is for

The paper carries only settled results, and the thesis carries the ones that fit
its argument. Everything else has to live somewhere or it gets rediscovered — or
worse, quietly reintroduced. This is that somewhere.

Three things it holds that nothing else does:

- **Derivations in full.** The paper states propositions; here they are proved
  with the algebra written out.
- **Retired experiments, with the reason.** Not just what was dropped, but what
  the analysis was that dropped it.
- **Pitfall boxes.** Concrete mistakes, with the measurement that exposed them:
  a boundary statistic quoted as if it were a bound, an M-step budget that
  under-converged the innovation shape, a stopping rule that accepted a decrease
  in the likelihood as convergence.

## Provisional results are allowed here

This is the one document where unfinished work belongs, which is why
`tools/check_paper.sh` does not gate it. Everything provisional says so.

Chapters `ch11-status` and `ch13-corrections` together hold the claim ledger:
what is settled, what is in progress, what was withdrawn and not replaced.
Check there before citing a number from an older draft.
