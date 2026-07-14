# Research/ index

Organisational map of `Score_Diffusion/Research/`. The directory has
accumulated multiple overlapping working notes, code packages, and
earlier explorations; this index sorts them into five tiers ordered
by current utility, so future readers know where to look first.

> **Convention.** The joint score is the central object throughout.
> Earlier toy models that computed the per-frame *marginal* score are
> superseded; the correction is documented in the project memory file
> `project_joint_score_correction.md` and recapped in
> `Score_Diffusion/CLAUDE.md`.

---

## Tier 1 — Canonical reference (read this first)

| Path | Role |
|---|---|
| `unified_document/` | Consolidated working note + reproducible code + 8 figures + 53-check audit. The single source of truth going forward. |
| `Relevant/` | The five session-summary PDFs that feed `unified_document/`: `Gaussian_session_summary.pdf`, `session_summary_K1_warmup.pdf`, `precision_lifecycle_summary.pdf`, `bp_session_summary.pdf`, `numerical_experiments.pdf`. |

The cuts applied to those five PDFs in the consolidation are listed in
`unified_document/README.md` and recapped in `main.tex`.

---

## Tier 2 — Source manuscripts (working sessions, not yet superseded)

These are the primary LaTeX/PDF working notes that the Tier 1 PDFs
were extracted from. Keep them for archival cross-reference.

| Path | Topic |
|---|---|
| `gaussian-lifecycle-session/` | Working note `precision_lifecycle_summary.pdf` + Python driver `ar1_precision_lifecycle.py` for the Gaussian benchmark figure pack. |
| `bp-session/` | Working note `bp_session_summary.pdf` + `bp_session_summary.tex`. The Convention-A BP derivation. (Code is rewritten from scratch in `unified_document/code/bp_score.py`.) |
| `Laplace-k1-k2/` | `laplace_K1_compendium.pdf`, `laplace_K1_benchmark_memo.pdf`, plus the K=1 manuscript and code. Source for Act II of the unified note. |
| `laplace_ar1_audit/` | Earlier audit suite for Laplace K=1: `code/run_checks.py`, `code/ar1_diffusion_utils.py`, audit `figures/`, `presentation/`. Useful pre-existing reference; the current audit lives at `unified_document/code/numerical_audit.py`. |
| `merged_core_notes/` | An earlier consolidation attempt: `merged_core_notes.tex` + `merged_core_notes.pdf` + `audit_report.md` + `source_map.md` + `figures_manifest.md`. `unified_document/` supersedes it. |

---

## Tier 3 — Code packages (drivers and figure scripts)

Self-contained code repositories that produced specific figure packs
or derivations. Useful when reproducing legacy figures or porting
specific helpers.

| Path | What it produces |
|---|---|
| `research_continuation_v2/` | Canonical figure pack `fig01--fig17` (cited in `numerical_experiments.pdf` Table 2-4); driver `code/generate_all_figures.py` + utilities `code/laplace_ar1_utils.py`. NOTE: figures fig15--fig17 use the K=1 proxy and are excluded from the unified document. |
| `research_continuation_package/` | An earlier version of the same package, with `long_derivation_note.tex` and `short_presentation_note.tex`. Largely superseded by `_v2/`. |
| `ar1_diffusion_project/` | Standalone Gaussian AR(1) note `ar1.pdf` + `main.tex` + `code/` + `figures/`. Cited from `precision_lifecycle_summary.pdf`. |
| `laplace_ar1_prism_package/` | Single-file note `laplace_ar1_ou_note.tex/.pdf` + curvature field figure. A prism-like view of the K=1 case. |

---

## Tier 4 — Earlier explorations (kept for archival reference)

These directories contain legitimate but superseded work. They are
useful for tracking the history of the project's understanding.

| Path | Status |
|---|---|
| `sigma_t_analysis/` | Earlier `Sigma_t` analysis: `structure_sigma_t.tex/.pdf` + `companion_explanations.tex/.pdf` + figures. Superseded by `precision_lifecycle_summary.pdf` and `unified_document/` Section 2. |
| `anatomy_sigma_t/` | Early anatomy of `Sigma_t`: `anatomy_sigma_t.tex/.pdf` + figures. Same content area as `sigma_t_analysis/`, both pre-date the precision-lifecycle synthesis. |
| `project/` | Three earlier sub-files: `file1_markov_loss/`, `file2_deterministic/`, `file3_laplace/` + a `build_all.sh`. Pre-dates the canonical session structure. |

---

## Tier 5 — Notes, meeting records, and auxiliary materials

| Path | Role |
|---|---|
| `Notes/` | Conceptual / reading notes: `Problem_formulation.pdf` (thesis framing, Q1-Q7), `2nd-meeting-notes.pdf` (board exercises), `Draft-AR1.pdf/.tex`, `Curiosity.pdf`, `Understanding.pdf`, `idee post-lettura note.rtf`, plus a Word draft of the joint PDF toy model. Background reading. |
| `Meeting-Mezard_07.05.2026/` | Latest meeting record: `.docx` and `.vtt` transcript; copies of two session summaries. |
| `conjecture15_test_note.md` | Markdown note explicitly documenting that the "approximate rank-2" Laplace K=2 Hessian conjecture is regime-dependent at quick resolution and pending HPC verification. The unified document does not assert it. |
| `numerical_experiments.tex` | Source of `Relevant/numerical_experiments.pdf`. |

---

## Recommended traversal

1. Read `unified_document/main.pdf` (12 pages, narrative arc Gaussian
   -> Laplace K=1 -> Laplace K>=2).
2. Run `unified_document/code/numerical_audit.py` to confirm 53/53
   PASS on the local environment.
3. If a specific Tier 1 PDF needs to be inspected directly, look in
   `Relevant/`.
4. Only descend into Tier 2-5 if you need (i) the source LaTeX of a
   working note, (ii) a legacy figure that was deliberately cut, or
   (iii) the historical evolution of a derivation.
