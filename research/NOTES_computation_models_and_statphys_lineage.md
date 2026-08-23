# Notes: models of computation, and the physics–ML lineage

Working notes, not thesis prose. Collected 2026-08-23 from three inputs: LeCun's
taxonomy of computation models, the Les Houches 2022 school editorial, and the
CFM presentation (`~/Downloads/cfm_physique_statistique_finale.pptx`).

The verdict is at the bottom. Short version: **one of these three is worth
putting in the introduction, and it is worth putting there properly.** The other
two are colour, and Chapter 2 already carries most of what they would add.

---

## 1. LeCun's four models of computation — the one that matters

Stated as given, lightly formalised:

| # | Form | Turing complete? | Examples |
|---|---|---|---|
| 1 | $y=f(x)$, $f$ a fixed number of sequential non-linear steps | **No**, unless $f$ is made infinitely wide | Feed-forward nets; AR-LLMs (fixed compute per token) |
| 2 | $z^{(k+1)}=g(z^{(k)},x)$, $y=f(z^{(K)})$, $K$ unbounded | Yes | Recurrent nets; **diffusion models** |
| 3 | $\hat z = \arg\min_z E(x,z)$, $y=f(\hat z)$ | Yes | Energy-based models; deterministic graphical models |
| 4 | $q = \arg\min_{q\in Q}\big[\langle E(x,z)\rangle_q - \tfrac1\beta H(q)\big]$, $y=f(\mathrm{sample}(q))$ | Yes | **Bayesian inference in probabilistic graphical models**; Friston's active inference |

Structural relations LeCun draws, all of which matter here:

- **3 is the $\beta\to\infty$ limit of 4** (zero temperature). 4 is the
  non-deterministic version of 3.
- **2 can precompute an approximate solution to 3 or 4.** This is *amortised
  inference*.
- 1 is not Turing complete because the computation per output is constant. An
  AR-LLM cannot solve a problem whose answer requires unbounded search (parity
  is the example) — and, more sharply, *cannot be trained to*, because you
  cannot backpropagate through the sampling/quantisation that produces tokens,
  which is why teacher forcing is used.

### Why this is directly about this thesis

This taxonomy names, precisely, the thing Chapter 9 measures and currently
describes only operationally.

- Sum–product on the posterior chain is **category 4**. It returns a
  distribution over latents, obtained by making a free energy stationary — and
  on a tree the variational family $Q$ is rich enough that the minimiser is
  *exact*, not an approximation. That is the whole reason the thesis has a
  computable target at all.
- The trained score network is **category 1**: fixed depth, fixed compute per
  evaluation, no mechanism to search.
- The diffusion sampler wrapping it is **category 2** — unbounded reverse
  steps — but each step's score is supplied by a category-1 network.
- Therefore: **the score network is an amortised approximation to what BP
  computes exactly.** Training it is category-2-precomputing-category-4, which
  is exactly LeCun's amortised-inference relation.

That reframes the headline comparison from "structured estimator beats network"
— which sounds like a horse race, and which the external review rightly said
was under-defended — into something with a name and a reason:

> On a family where the category-4 computation is tractable, we can measure what
> amortising it into a category-1 approximator costs. That cost is the
> 7–20× (against a generic MLP) or 1.8–5.5× (against a tuned convolution) of
> Chapter 9.

It also explains the *shape* of the exp_12 result rather than merely reporting
it. The convolution recovers nearly all the gap at $n_{\text{seq}}=32$ and less
and less as data grows: amortisation is cheap when there is little to amortise,
and the exact inference pulls away precisely as the posterior it computes
becomes better determined. A fixed-compute approximator has no mechanism to
spend more effort on a harder posterior; BP does, because $K$ is the chain
length and the recursion runs to completion regardless.

And it sharpens the paper's own open question. "Which architectures exploit
Markovianity, and at what depth" is, in this language, "how deep must a
category-1 approximator be before it amortises a category-4 computation well",
and the U-Net/transformer-as-BP results already cited (`mei2024unet`,
`garnierbrun2024transformers`) are answers to exactly that question.

### Citable anchors (the tweet is not one)

- `lecun2006tutorial` — **already in the bibliography and already used in
  Appendix F.** This is the right anchor for categories 3 and 4; the appendix
  already discusses the energy-as-primitive framing.
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022, OpenReview) —
  the position paper where the inference-by-optimisation architecture is laid
  out. **Verify the exact title/venue before citing.**
- LeCun's Les Houches 2022 lecture, *From machine learning to autonomous
  intelligence* (listed in the school editorial below, video linked).
- Amortised inference as a term predates this framing and has its own
  literature; if the thesis uses the word it should cite that rather than a
  taxonomy blog post.

**Do not cite the tweet.** The taxonomy is a useful organising device that can
be stated in the thesis's own words with `lecun2006tutorial` for the EBM half;
attributing a four-way taxonomy to a social-media post in a Bocconi thesis is a
weakness a viva examiner would notice.

---

## 2. Les Houches, and the physics–ML lineage

From the editorial to the Les Houches 2022 special issue
(*J. Stat. Mech.* 2024 101001, https://iopscience.iop.org/article/10.1088/1742-5468/ad4e2a):

Lecturers relevant here:
- **Marc Mézard** (Bocconi) opened the school with *Belief propagation,
  message-passing and sparse models* — i.e. the thesis's own machinery, taught
  by the thesis's own supervisor, at the canonical venue for this material.
- **Giulio Biroli** — high-dimensional non-convex landscapes and gradient-descent
  dynamics. Already cited in the thesis (`biroli2023`, `biroli2024dynamical`).
- **Yann LeCun** — *From machine learning to autonomous intelligence*.
- Monasson (replica), Srebro, Solla, Montanari, Sompolinsky, Barak, Jordan,
  Bach, Bahri & Hanin.

Lineage anecdotes from the same editorial, all quotable:
- LeCun, *Quand la machine apprend*: "Ma vie professionnelle bascule réellement
  en 1985 lors d'un symposium aux Houches."
- Hinton, on the early days of neural nets, crediting "one really smart
  physicist, Elizabeth Gardner".
- Isabelle Guyon on the invention of SVMs: Mézard and Krauth's *minover*
  optimal-margin algorithm "attracted my attention […] it was not until I joined
  Bell Labs that I put things together".
- Parisi's 2021 Nobel (replica method, explicitly including its use in machine
  learning per the committee); Talagrand's 2024 Abel Prize for the mathematics
  behind replica results.

**Relation to the thesis as it stands.** Chapter 2 §"From the Ising model to
neural networks" already carries Hopfield → Boltzmann machines → the 2024 Nobel,
already cites `gardner1988space` and `mezard2017meanfield`, and already makes
the cavity/TAP → message-passing descent argument explicitly ("historical
descent, not analogy"). The editorial does not add a claim; it adds *evidence*
for one already made. Its best use is one footnote, not a section.

The Mézard-taught-BP-at-Les-Houches point is the exception: it is a genuinely
apt one-line justification for why this thesis reaches for message passing
rather than a network, and it costs a sentence.

---

## 3. CFM and finance — tangential, and should stay so

From the deck (`cfm_physique_statistique_finale.pptx`), which is a course
presentation, not thesis material:

- CFM founded 1991 in Paris by Jean-Pierre Aguilar; Jean-Philippe Bouchaud
  founded Science & Finance 1994; merged 2000; ~$18bn AUM (2025).
- Random matrix theory for separating true correlations from sampling noise:
  Laloux, Cizeau, Bouchaud & Potters, *Noise dressing of financial correlation
  matrices*, PRL **83** 1467 (1999).
- Fat tails as the recurring quantitative theme.
- Les Houches 2006 *Complex Systems*, ed. Bouchaud, Mézard & Dalibard — the
  same school, same editors, the interdisciplinary framing.

**Verdict: keep out of the thesis.** The link is real at the level of "statistical
mechanics of many interacting noisy components", but the thesis is about exact
inference on a Markov chain under a diffusion channel. Finance would be a third
domain introduced and then not used, and the external review's central
structural criticism was *precisely* that the document reads as two theses
stapled together. Adding a third strand runs directly against the 100-page
target and against the "one question, asked three times" spine.

One exception worth considering: **fat tails**. The thesis's whole point is
variance-matched innovation families that differ only beyond second moments, and
Laplace/Student innovations are exactly the heavy-tailed case. A single sentence
in the introduction's motivation — that non-Gaussian tails are the practically
consequential departure in real sequential data, financial time series included,
with the PRL as the citation — would be earned rather than decorative. Anything
more is scope creep.

---

## Verdict: what should actually go in the introduction

**Recommended (worth the space, ~half a page in Ch1 §"Positioning and gap"):**

1. LeCun's taxonomy, stated in the thesis's own words, used to place the two
   arms of the R4 comparison: exact BP is inference-by-free-energy over a
   distribution, the score network is a fixed-compute amortisation of it, and
   the thesis measures the cost of that amortisation on a family where the
   exact object is computable. Cite `lecun2006tutorial`; verify and add the 2022
   position paper if it is to be leaned on.
2. One sentence noting that the amortisation framing predicts the shape of the
   exp_12 result — the gap widening with $n_{\text{seq}}$ — rather than only
   describing it.

**Marginal (one footnote at most):**

3. Mézard's Les Houches BP lecture, as the disciplinary home of the method.

**Reject:**

4. The CFM/finance material, except possibly one clause on heavy tails.
5. Any expansion of the Hopfield/Gardner/Nobel lineage — Chapter 2 has it, and
   that chapter was just cut from 25 to 21 pages to reach the 100-page target.
   Re-expanding it would undo work done for a reason.

**Where it goes, and what it costs.** Chapter 1 is 9 pages and the main body is
103 against a 100 target, so item 1 has to be paid for.

I first assumed §"Positioning and gap" had slack. Checked: it does not. It is
three tight strands — statistical physics of learning machines, inference on
chains, message passing with continuous variables — each ending on what that
strand does *not* answer, closing with a one-sentence gap statement. There is
nothing to trim without losing an argument.

More useful is what that section reveals: **there is no strand on neural score
approximation at all.** The three strands motivate the *analytical* contribution
(an exactly solvable account of the joint score) and the gap statement is about
exactly that. The network comparison of Chapter 9 is currently motivated
nowhere in the introduction — it arrives as an experiment rather than as an
answer to a stated question.

So the taxonomy is not a decoration to be squeezed in; it is the missing fourth
strand, and it earns its space by fixing a real hole:

> **Amortising inference into a network.** Diffusion sampling is iterative, but
> the score at each step comes from a fixed-compute feed-forward map trained to
> approximate it — an amortisation of an inference problem into an
> approximator that cannot search. Whether that amortisation is cheap or
> expensive is unmeasurable in general, because the amortised target is
> unavailable. It is measurable here.

That paragraph both motivates Chapter 9 and states the taxonomy's content
without a table. Cost: roughly half a page in Ch1, to be recovered from Chapter
2 (still 21 pages) or Chapter 5 (17 pages, against the review's 12–14
allocation) — not from this section.
