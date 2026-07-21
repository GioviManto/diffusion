# Editorial decision register

Standing contract: 2026-07-21 author instructions (senior-editor prompt +
complaint list). Source of truth: the newest attached PDF (= repo v2 source,
verified identical in content). This register records structural and
terminology decisions; see CHANGELOG.md for what moved, CORRECTNESS.md for
claim status, FIGURES.md for the figure inventory.

## Structure (v3, decided 2026-07-21)

1. Introduction (short; problem reached within two pages)
2. Model, score identities, research context, research development
   (merges old Ch2 + Ch3 + Ch4, cut to the minimum used later)
3. The Gaussian chain: exact score, belief propagation, Kalman, locality
   (merges old Ch5 + the Gaussian parts of Ch7; complete in one place)
4. Beyond Gaussianity: Laplace innovations and the Gaussian-message
   approximation (merges old Ch6 + the experiments of old Ch7)
5. Discussion and conclusions
- App A: Gaussian identities
- App B: AMP/TAP and the bulk fixed point (kept: 100% independently
  re-verified 2026-07-21, see CORRECTNESS.md)
- App C: Reproducibility

## Standing rules

- Research questions: the four modest RQs fixed by the author (2026-07-21).
  No claims about universal theories, general Markov processes, or practical
  architectures.
- "Score–posterior identity", never "Tweedie identity" (one attribution
  parenthetical allowed at the identity's statement).
- Honest complexity: BP does O(K) message *updates* on the chain; O(K)
  *cost* only under a closed finite-dimensional message family.
- Grid BP is a numerical approximation of the functional-message recursion;
  it is called an "exact reference" only inside its validated regime, with
  the validation stated.
- Architecture implications are possible readings of a controlled model,
  never established conclusions about trained networks.
- Every claim classified: exact / assumption-conditional / numerical /
  interpretation / limitation / open.
- Contribution verbs: derive, characterise, establish under assumptions,
  show algebraically, compare, measure, observe in the tested regime.
- Style: no "---" decoration, em dashes rare, no rhetorical questions, no
  chapter-opening drama, no "two movements", "mother theorem", "punchline",
  "miracle", "where closed forms die", "old-fashioned methodology".
- One explanation per concept, in its natural place; no ping-pong
  cross-references; a later chapter never completes an earlier derivation.
- Figures: matplotlib, white background, vector PDF, one figure per
  distinct result, caption states model/parameters/statistic/observation.
- Never reintroduce deleted material silently (log in CHANGELOG.md).
- Never invent references, results, or numbers; supervision remarks are not
  evidence and are not quoted verbatim in the body.
- Preliminary research (toy models, the joint-score correction, the
  continuous-message bottleneck) appears once, in Ch2's research-development
  section, in the form: initial question -> attempt -> difficulty ->
  revised question -> final method.

## Notation (unchanged from v2)

a (clean frames), x (noisy frames), K frames, alpha coupling,
sigma_eta^2 = 1 - alpha^2, mu = e^{-t}, Delta_t = 1 - e^{-2t},
Sigma_0/Q_0 clean covariance/precision, Sigma_t/Q_t noisy, J posterior
precision, J_d/beta bulk parameters, q bulk decay rate, S(x,t) score.
