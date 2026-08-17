# The frozen batch — job ledger

Every number that may enter a document comes from a job listed here. All of them
run `hpc/bocconi_frozen.sbatch`, which imports `experiments/frozen_config.py`;
`tools/check_paper.sh` fails if any experiment sets rho, the grid or an
iteration count locally.

Sorted newest first. "Usable" means the result may be cited somewhere; where it
may be cited is the last column, and `—` means nowhere yet.

| Job | Task range | What | State | Usable | Cited in |
|---|---|---|---|---|---|
| `629962` | 90–105 | Rung 4a, matched convergence budget (8 seeds × 2 size groups, cap 3000, per-sequence tolerance) | RUNNING | pending | — |
| `629193` | 80–87 | Rung 4a v2 — one species, 600 steps, 16 seeds | COMPLETED 8/8 | partly | compendium §Rung 4a |
| `629091` | 60–75 | E9 sample efficiency, validation-selected EM stopping | COMPLETED 16/16 | not yet | — (protocol still asymmetric) |
| `628943` | 40–55 | E9 sample efficiency at 400 EM iterations | COMPLETED 16/16 | not yet | — |
| `628942` | 30–33 | Rung 4a v1 — **defective**, two species scored at one psi | COMPLETED 4/4 | **no** | compendium, as a correction |
| `628601` | — | shape convergence | COMPLETED 2h29m | yes | compendium |
| `628600` | — | clean-vs-channel rates | COMPLETED 4h01m | yes | compendium |
| `628571` | — | E9 sample efficiency, 16 seeds | COMPLETED | yes | paper §efficiency (with caveat) |
| `628434_5` | 5 | Rung 4a, first attempt | **TIMEOUT** 6h00m | no | — |

## The three that are worth knowing about

**`628434_5` — killed by the wall having written nothing.** 960 gradient fits at
~0.2 s per step is roughly ten hours, and the experiment writes once at the end.
Fixed by sharding on seed, which is exact rather than approximate because
`part_potential` reseeds per seed.

**`628942` — completed and produced nothing usable.** `part_potential` generated
two species at ±omega while `fit_potential` scores every trajectory at the single
scalar psi it is handed, so half the data was evaluated under the wrong rotation.
The estimator escaped by collapsing the ring — at r\* → 0 the ring is
rotation-invariant and so pays nothing for a wrong psi — which pinned r\* to the
lower clip in 51.2% of cells and made the *marginal* arm appear to beat the joint
arm. Pinned both ways by
`tests/test_ring.py::test_mixed_species_data_collapses_the_ring_under_a_single_psi`.

**`629193` — fixed the estimator, but not the comparison.** Clip pinning fell to
4.8% and the joint arm recovered its expected ordering, so the joint arm's
n-scaling is reportable. The joint-versus-marginal *margin* is not: at t ≤ 1 the
marginal arm hit the 600-step cap in 95% of cells against the joint arm's 57%, so
the two arms were not converged to the same standard. Measured directly with the
plateau test disabled (n = 512, t = 0.2216): the joint arm is unchanged to eight
significant figures from step 600, while the marginal arm's lambda error keeps
falling — 0.482 at 600, 0.372 at 1200, 0.335 at 2400. `629962` reruns it under a
per-sequence tolerance, which is what the stopping rule should have been all
along: the old absolute tolerance was eight times stricter at n = 4096 than at
n = 512, i.e. it varied along the very axis the rung reports its scaling on.

## Not yet run

- **Network early stopping.** Until the network's training length is chosen the
  same way EM's stopping point now is, the efficiency comparison tunes one arm
  and not the other, and the table stays behind a `\needsdata` marker.
