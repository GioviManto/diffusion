# Correctness issue log

Status values: VERIFIED (independent check passed), REMOVED (could not be
verified or not fit for purpose), QUALIFIED (kept with weakened wording).

## Resolved 2026-07-21 (v3 pass)

| Claim | Status | Evidence |
|---|---|---|
| Bulk variance V = 1/sqrt(Jd^2-4b^2); (J^-1)_{i,i+d} = q^d V | VERIFIED | fresh numpy check vs brute-force inverse of K=401 tridiagonal Toeplitz, err < 1e-10 across (alpha,t) grid; repo audit 72/72 |
| Cavity fixed point lambda*, recombination sqrt(Jd^2-4b^2) | VERIFIED | fresh iteration check, err < 1e-12 |
| AMP variance root, existence iff Jd >= 2*sqrt(2)|b| | VERIFIED | fresh damped iteration matches closed form < 1e-10; divergence confirmed past boundary |
| t_c(alpha) closed form; alpha_c = sqrt(2)-1 | VERIFIED | at t=t_c the boundary Jd = 2*sqrt(2)|b| holds to 1e-10; sign of g flips at alpha_c |
| Weak-coupling law V_AMP - V = 2b^4/Jd^5 + O(.) | VERIFIED | ratio -> 1 as alpha -> 0 (1.004 at alpha=0.02, 1.001 at 0.01) |
| AMP/BP/MF means all solve Jm=h (score closure-independent) | VERIFIED | fresh damped iteration vs solve, err 3e-15 |
| Band-fill law, no 1/(d-1)! factor | VERIFIED | rel err ~ 4e-4 at t=1e-5 for d=1..4 (d=5 at t=1e-5 hits float64 cancellation; verified at t=1e-3 within its asymptotic band); factorial version rejected (99%-42000% off) |
| Repo audits | VERIFIED | 72/72 and 55/55 re-run on build machine 2026-07-21 |
| Laplace closure numbers (0.39 @ t=0.02 ... 7.6e-5 @ t=2.4; cosine >= 0.92/0.78) | VERIFIED (2026-07-14 re-check from CSVs) | kept with regime qualifications |
| Truncation comparison 12.3% vs 21.0% at t=0.05 | VERIFIED (audit) | moved to Gaussian chapter (it is a Gaussian-model experiment) |
| "Machine-precision agreement" BP vs matrix | QUALIFIED | reported with parameter grid, tolerances, and independent implementation stated; code in repo |
| Laplace K=2 Hessian approx rank-2 conjecture | REMOVED from claims (kept as negative result/limitation) | regime-dependent under tests |
| Architecture implications (local/convolutional heads) | QUALIFIED | stated as readings of the Gaussian model only, RQ4 wording made modest |

## Open

- Non-Gaussian locality laws: only structurally delimited; stated as open.
- Reverse-dynamics error propagation: not claimed; listed as future work.

## v4 additions (2026-07-22)
No new numerical or mathematical claims introduced. New related-work
statements are cited summaries; each new bib entry's title/authors/arXiv
id was extracted directly from the PDF in sources/ before citing
(achilli2026speciation 2602.04404, garnierbrun2026biased 2603.03469,
bonnaire2025memorize 2505.17638, sclocchi2025phase 2402.16991,
mei2024unets 2404.18444, holderrieth2025flow 2506.02070,
krzakala2024statphys lecture notes 2024, achilli2026thesis Bocconi PhD
thesis, lai2025principles 2510.21890, ronneberger2015unet MICCAI 2015).
Restored v2 chapters (2, 3, 4) reuse previously audited derivations
verbatim up to style. Ch. 5/6 contain standard textbook material
(cited) and project history (from the archived Score_Diffusion repo and
board photos); the board content is used as narrative source only, not
quoted or presented as evidence. The two restyled appendix figures
recompute the same audited formulas; the new toy-model figure evaluates
elementary Gaussian-mixture identities on a grid.
