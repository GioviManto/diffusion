# Source Audit — what was used, what was archived, and why

Consolidation date: 2026-07-14; revision v2 on 2026-07-19. Nothing was
deleted; everything not listed as signal lives intact in
`~/Code/Thesis/archive/` (outside the git repo).

## Sources added in revision v2 (2026-07-19)

| Source | Used for | Location |
|---|---|---|
| `Things-to-change-thesis.pdf` (author feedback) | drives the whole revision | copy in `meetings/` (gitignored) |
| `Desktop/BOCCONI/.../STAT-MECH/SM_Paper_Notes.pdf` (59-pp reading notes on the cascade paper, Giovanni, Nov 2025) | thesis Ch. 4 cascade section (restored to full depth in v4; was one paragraph in v3) | external, Desktop (course folder) |
| `Mantovani_Slides_SM.pdf` + `Script.pdf` (course presentation for Mézard's 41002) | thesis Chs. 2 and 4 (statistical mechanics and cascade chapters) | external, Desktop (course folder) |
| Bachtis–Biroli–Decelle–Seoane, NeurIPS 2024 (cascade paper); LeCun et al. 2006 EBM tutorial; Mézard 2017 PRE; Hopfield 1982; AGS 1985; Ackley–Hinton–Sejnowski 1985; Smolensky 1986; Hinton 2002/2006; Gardner 1988; Saxe 2014; Decelle 2017; Tubiana–Monasson 2017; flow-model papers (Rezende 2015, Papamakarios 2021, Chen 2018, Lipman 2023, Albergo 2023) | new literature chapters; all verified published venues | cited in `thesis/references.bib` |

The unofficial Zdeborová–Krzakala 2021 EPFL lecture notes were removed
from the bibliography in v2 (uncited after the rewrite; the published
review `zdeborova2016statistical` covers the material).

## Signal (in this repo)

| Source (original location) | Now at | Why it is signal |
|---|---|---|
| `~/Code/gaussian-bp-diffusion` (repo, 31-pp note, 3 notebooks, 72-check audit) | `research/gaussian-bp/` | Canonical Gaussian chain + BP/AMP results; audit re-run 72/72 on 2026-07-14 |
| `…/gaussian-bp-diffusion/bp-from-scratch` | `research/bp-from-scratch/` | Independent equation-by-equation BP derivation from the 5 Jun call; truncation experiments (12.3% vs 21.0%) |
| `Desktop/Diffusion/Score_Diffusion/Research/unified_document` | `research/unified-note/` | Consolidated working note; Laplace Acts II–III; audit re-run 55/55 |
| `Desktop/Belief-prop/bp_markov_informed_scores` | `research/nongaussian-bp/` | Grid BP vs Gaussian-projected BP; closure-error and cosine numbers re-verified from CSVs on 2026-07-14 |
| `Desktop/Belief-prop/bp_gaussian_ar1_package` | `research/gaussian-ar1-bp/` | Package behind the Pages site shared with Jérôme (BP = matrix score) |
| `Desktop/Belief-prop/bp_markov_diffusion_gaussian_approx_package` | `research/bp-generalization/` | General non-Gaussian BP formulation sketches (`general-BP*.jpeg`) + code |
| `Score_Diffusion/Research/Relevant/` (5 session PDFs) + `Notes/Problem_formulation.pdf` | `research/session-summaries/` | Source working notes feeding the unified note; thesis framing Q1–Q7 |
| `Score_Diffusion/Experiments`, `Desktop/Diffusion/figures`, `modelling.ipynb` | `research/initial-experiments/` | Initial diffusion-setup experiments — start of the research narrative |
| `Desktop/Diffusion/Thesis-Notebook` (24 scans) | `research/notebook-scans/` | Handwritten derivations |
| `BP continuous case.docx`, `16-2-2025 call.rtf`, transcripts, board sketches, Mézard meeting records | `meetings/` (**gitignored**, local only) | Supervision record; the 5 Jun call defines the research programme |
| `main-sources/`, `Papers/`, `BF-theory/` | `sources/` (**gitignored**, local only) | Copyrighted PDFs; kept out of the repo |
| `Official-Thesis/guide.pdf`, `Guida_tesi_LM_ENG…` | `thesis/official-guides/` | Binding format rules |

## Noise (archived intact at `~/Code/Thesis/archive/`)

| Item | Why archived, not used |
|---|---|
| `thesis_part1_expanded.pdf` (78 pp. draft) | Rejected by author: wrong references, doesn't follow guide.pdf, uneven depth. Superseded by the new `thesis/`. Not used as a source for any claim. |
| `Score_Diffusion/Toy-models` TM1–TM6 | Computed per-frame **marginal** scores — conceptually superseded by the joint-score correction (Mézard meeting 2026-03-12). Kept for project history. |
| `Score_Diffusion/Research/` Tier 2–5 (sigma_t_analysis, anatomy_sigma_t, merged_core_notes, research_continuation*, project/, laplace_ar1_audit, Laplace-k1-k2 …) | Working sessions superseded by `unified-note/` (see its RESEARCH_INDEX.md tiering). `precision_lifecycle_summary.pdf` contains the known band-fill error (eq. 12). |
| `iap-diffusion-labs` | External MIT course labs — reference material, not project output. |
| `venv/`, `.venv/` trees | Environments; recreate from `requirements.txt`. |
| `Claude-chats.txt`, briefings, HTML slides/labs | Session artefacts; superseded by this repo's docs. |
| `bayesian_diffusion_amp_textbook.pdf`, `mathematical_methods_handbook.pdf`, `main.pdf` (root) | Generated study material, not citable sources; superseded by the real bibliography. |
| `bootstrap-dataset.jsonl`, misc screenshots/SVG/PNG at root | Loose artefacts, preserved in archive. |

## Reference hygiene

- `Generative_diffusion_updated_notes_MM.pdf` (Mézard's unofficial lecture
  notes) is **never cited** in the thesis; published works
  (Biroli–Mézard 2023 JSTAT; Biroli–Bonnaire–De Bortoli–Mézard 2024
  Nat. Comm.; Mézard–Montanari 2009) are cited instead.
- Every bibliography entry was written from verified bibliographic data;
  no entry was copied from the rejected draft.
- The Bocconi author–date style (writing guide, Attachment 2) is applied
  via natbib/apalike; online resources carry last-access dates.

## Known repository duplication (deliberate)

`research/*` contains cleaned copies (no venvs, no git dirs, no aux files)
of material whose originals — including full git histories — are preserved
in `archive/`. The duplication is the price of a self-contained repo and
costs ~55 MB.

## GitHub state (as of consolidation)

Thesis-related remotes existing before consolidation:
`GioviManto/Score_Diffusion`, `GioviManto/gaussian-bp-diffusion`,
`GioviManto/bp-gaussian-ar1-diffusion` (all public; the last one's Pages
site is linked in the email Jérôme received — **do not delete it before
he has the new link**). The consolidated repo replaces all three going
forward; archiving (not deleting) the old ones on GitHub is recommended,
and is a user action.
