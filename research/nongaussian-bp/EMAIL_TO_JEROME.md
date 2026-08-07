# Draft email to Jérôme — not sent

Fill in the five `<...>` links from the pushed branch before sending. They are listed at the
bottom of `FINAL_REVISION_REPORT.md` once the push completes.

---

**To:** Jérôme Garnier-Brun
**Subject:** Quick feedback on the BP–diffusion update before the weekend

---

Dear Jérôme,

Apologies for the short notice — I know tomorrow is Friday and this arrives without much warning.
I have spent this week reworking the BP–diffusion material fairly thoroughly, and I would be
grateful if you had time for even a brief look before the weekend, since I am freer than usual
over the next few days and would rather spend that time on the things you think matter most than
guess at them.

A note on how the work was done, in the interest of transparency. I used Claude and ChatGPT as
assistants — to explore formulations, debug code, organise the experiments and improve the
write-up — and the heavier numerical runs went through the Bocconi HPC system. I have gone back
through the mathematical claims and checked them against the implementation and the committed
outputs rather than against my memory of what the code did; that pass is written up as a separate
audit, and it did turn up things that needed correcting. None of that is a substitute for your
reading it, which is why I am asking.

The short version of where it stands:

- Posterior inference on the noised chain remains a chain, and sum-product on it is exact at the
  level of functional messages. The grid implementation is exact for the discretised model, and
  I now measure truncation and quadrature error separately rather than quoting one sweep.
- The unknown kernel can be fitted from noised sequences in the parametric setting we set up.
  Both the autoregressive coefficient and the innovation density are estimated — ρ is initialised
  at 0.3 against a truth of 0.85 and comes back at 0.850.
- The structured estimator is strong pointwise: 9–14× lower relative denoising error than the
  networks I trained, and 2–4× against a locality-respecting convolution.
- Pointwise and generative rankings can disagree when the innovation model is too restricted.
  This holds across four non-Gaussian families at matched covariance, and — the useful control —
  it vanishes on the Gaussian chain, where a four-component mixture is exact.
- Increasing mixture capacity closes it, and the run that was outstanding has now returned:
  pointwise error and generated-tail fidelity both improve monotonically in the component count,
  so in the range tested the two axes do not trade off.

Some checks remain open, and I have kept them visible rather than buried: no validation split for
the neural model selection, no measurement of marginal likelihood or runtime against capacity, and
identifiability is assumed rather than proved.

The documents are separate so you can read only as much as you have time for:

- Repository and branch: `<BRANCH_URL>`
- Executive summary, two pages — this is the one to read if you have ten minutes:
  `<EXEC_SUMMARY_PDF_URL>`
- Answers and questions for the advisors, which reconstructs and answers the questions from our
  call: `<QA_PDF_URL>`
- Main note, ten pages plus appendices: `<PAPER_PDF_URL>`
- Technical compendium, the long-form derivations: `<COMPENDIUM_PDF_URL>`

The questions where your judgement would help most:

1. Whether the framing should be "transition-kernel estimation" or the narrower "innovation law".
   ρ genuinely is estimated, so I kept the stronger phrasing, but the kernel is constrained to the
   linear form K(a'|a) = φ(a' − ρa) and I am not sure that restriction should sit in the title.
2. Whether to learn ρ and the initial law μ jointly next. The E-step already computes the site-one
   statistic and throws it away, so it is a small change, but it costs the sufficiency argument as
   currently stated.
3. Which of the remaining experiments to prioritise — my order is validation-based model
   selection, then a direct locality measurement, then the strictly stationary rerun.
4. Whether the generation result belongs in the main text now that the capacity sweep resolves it,
   or whether it is better placed as an appendix ablation.
5. How central the locality / receptive-field direction should remain. It is the question I answer
   least well at the moment, and I would rather know now whether it is worth investing in.

No pressure at all if Friday does not work — anything you can send whenever suits you is useful,
and I will keep going on the items above in the meantime.

Best,
Giovanni
