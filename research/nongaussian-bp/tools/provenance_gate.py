"""Refuse to typeset a table whose inputs cannot be attributed to a revision.

WHY THIS EXISTS (round-two review item 4)
-----------------------------------------
Two defects motivated this, and they are the same defect at opposite ends.

At the producing end, a run could record a commit that did not describe its own
source. `outputs/exp_12_scaled` names commit 286b305, whose experiment file has
no `eff_seed0` setting, while the recorded command passes `--set eff_seed0=0`
into an override function that exits on unknown keys. `hpc/deploy_clean.sh`
closes that by shipping a hashed `git archive`.

At the consuming end, nothing checked. `make_tab_structured.py` globbed CSVs,
never opened a params file, and -- worst of the three -- FELL BACK TO THE PILOT
when it found fewer than eight scaled seeds. A generator that silently
substitutes weaker data for missing data produces a table that looks finished.
The failure is invisible in the PDF, which is the only artefact anyone reads.

So the contract here is refusal, not repair. Every check below fails the build
rather than returning something usable, because each one has a mode where the
wrong answer still typesets:

    dirty source      the numbers are not reproducible
    mixed commits     two programs' results averaged into one cell
    mixed configs     two protocols averaged into one cell
    duplicate seeds   one seed silently double-weighted
    missing cells     a seed's schedule average taken over a different schedule
    extra cells       a stale run leaking into a certified directory

`missing cells` is the subtle one. If seed 3 lacks the t=3.0 column, its mean
is over eleven levels while every other seed's is over twelve, and the pooled
number is a weighted average of two different estimands -- with no NaN anywhere
to reveal it.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path


class ProvenanceError(SystemExit):
    """Fails the build. Deliberately not catchable as a plain Exception."""


def _fail(title: str, detail: str, remedy: str) -> None:
    raise ProvenanceError(
        f"\nPROVENANCE GATE: {title}\n\n{detail}\n\nWhat to do: {remedy}\n"
    )


def load_params(dirs) -> list[tuple[Path, dict]]:
    """Every params_*.json under the given directories, with its path."""
    found = []
    for d in dirs:
        d = Path(d)
        for p in sorted(d.rglob("params_*.json")):
            try:
                found.append((p, json.loads(p.read_text())))
            except json.JSONDecodeError as exc:
                _fail("unreadable params file", f"{p}: {exc}",
                      "the run was interrupted mid-write; rerun that shard")
    if not found:
        _fail(
            "no params files",
            f"searched: {[str(x) for x in dirs]}",
            "these outputs predate provenance stamping, or the path is wrong. "
            "A table cannot be certified from CSVs alone.",
        )
    return found


def require_clean(params, *, allow_legacy: bool = False) -> None:
    """Every contributing run came from a clean, digest-stamped deployment."""
    dirty, undigested = [], []
    for path, d in params:
        if d.get("git_dirty"):
            n = len([x for x in str(d["git_dirty"]).replace(";", "\n").splitlines() if x.strip()])
            dirty.append(f"{path.parent.name}: {n} uncommitted path(s)")
        if not d.get("source_archive_sha256"):
            undigested.append(f"{path.parent.name}: no source-archive digest")

    if dirty:
        _fail(
            "outputs came from a dirty source tree",
            "\n".join(f"  {x}" for x in dirty[:12])
            + (f"\n  ... and {len(dirty) - 12} more" if len(dirty) > 12 else ""),
            "rerun via hpc/deploy_clean.sh from a committed tree. The recorded "
            "commit does not reconstruct these numbers.",
        )
    if undigested and not allow_legacy:
        _fail(
            "outputs predate the source-archive digest",
            "\n".join(f"  {x}" for x in undigested[:12])
            + (f"\n  ... and {len(undigested) - 12} more" if len(undigested) > 12 else ""),
            "these were deployed by the old stamp+rsync path, which is the one "
            "that mis-stamped exp_12. Redeploy with hpc/deploy_clean.sh, or pass "
            "allow_legacy=True and say so in the caption.",
        )


def require_single(params, field: str) -> str:
    """All runs agree on `field`. Returns the shared value."""
    seen: dict[str, list[str]] = {}
    for path, d in params:
        seen.setdefault(str(d.get(field, "")), []).append(path.parent.name)
    if len(seen) > 1:
        detail = "\n".join(
            f"  {v or '(empty)'}  <- {', '.join(sorted(who)[:6])}"
            + (f" +{len(who) - 6}" if len(who) > 6 else "")
            for v, who in sorted(seen.items())
        )
        _fail(
            f"runs disagree on {field}",
            detail,
            "these are different programs or different protocols. Pooling them "
            "averages two estimands. Regenerate the odd ones or table them "
            "separately.",
        )
    return next(iter(seen))


def require_complete(rows, axes: dict, *, key=lambda r: r) -> None:
    """The rows cover exactly the declared Cartesian product, once each.

    `axes` maps a column name to the values it must take, e.g.
        {"seed": range(16), "n_chains": [32,128,512,2048], "method": [...]}
    """
    names = list(axes)
    want = {tuple(c) for c in product(*(list(axes[n]) for n in names))}
    got: dict[tuple, int] = {}
    for r in rows:
        r = key(r)
        try:
            cell = tuple(type(next(iter(axes[n])))(r[n]) for n in names)
        except (KeyError, ValueError, TypeError):
            continue
        got[cell] = got.get(cell, 0) + 1

    missing = sorted(want - set(got))
    extra = sorted(set(got) - want)
    dupes = sorted(c for c, k in got.items() if k > 1)

    def _show(cells):
        return "\n".join(
            "  " + ", ".join(f"{n}={v}" for n, v in zip(names, c)) for c in cells[:10]
        ) + (f"\n  ... and {len(cells) - 10} more" if len(cells) > 10 else "")

    if missing:
        _fail(
            f"{len(missing)} of {len(want)} expected cells are missing",
            _show(missing),
            "a seed whose schedule is short averages over a different set of "
            "levels than its peers, and nothing downstream shows a NaN. Rerun "
            "the missing shards, or narrow the declared axes and say so.",
        )
    if dupes:
        _fail(
            f"{len(dupes)} cells appear more than once",
            _show(dupes),
            "a shard was merged twice, or two runs wrote the same seed. The "
            "duplicated seed carries double weight in every aggregate.",
        )
    if extra:
        _fail(
            f"{len(extra)} cells are outside the declared design",
            _show(extra),
            "a stale or exploratory run is in a certified directory. Move it "
            "out, or widen the declared axes deliberately.",
        )


def certify(dirs, rows, axes, *, allow_legacy=False) -> dict:
    """The whole gate. Returns the provenance a caption should quote."""
    params = load_params(dirs)
    require_clean(params, allow_legacy=allow_legacy)
    commit = require_single(params, "git_commit")
    cfg = require_single(params, "resolved_config_hash")
    require_complete(rows, axes)
    return {
        "commit": commit,
        "config_hash": cfg,
        "archive_sha256": require_single(params, "source_archive_sha256"),
        "n_runs": len(params),
        "n_rows": len(rows),
    }
