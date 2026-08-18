"""Shared experiment scaffolding: paths, argument parsing, provenance."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def experiment_parser(name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=name, description=description)
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced settings for a fast smoke run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / name,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--only", default=None,
        help=(
            "Comma-separated part names to run (default: all). Parts are "
            "independent and write disjoint CSVs, so a scheduler can run them "
            "as parallel array tasks into one --output-dir with no merge step. "
            "Use --list-parts to see the names."
        ),
    )
    parser.add_argument(
        "--list-parts", action="store_true",
        help="Print the part names this experiment defines, then exit.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Override a setting, repeatable, e.g. --set n_rep=64 "
            "--set sizes='(64,256,1024,4096)'. VALUE is parsed as a Python "
            "literal. Overrides land in params.json, so a scaled-up run stays "
            "self-describing."
        ),
    )
    return parser


def apply_overrides(settings: dict, assignments) -> dict:
    """Apply `--set KEY=VALUE` assignments to a settings dict.

    Unknown keys are rejected rather than silently added: a typo in a job
    script would otherwise run the default configuration on the cluster and
    look like a completed experiment.
    """
    import ast

    updated = dict(settings)
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in updated:
            raise SystemExit(
                f"--set: unknown key {key!r}. Known keys: {sorted(updated)}"
            )
        try:
            updated[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"--set {key}: cannot parse {raw!r} ({exc})") from exc
    return updated


def select_parts(parts: dict, only: str | None) -> dict:
    """Filter an ordered {name: callable} mapping by a --only specification."""
    if not only:
        return parts
    wanted = [p.strip() for p in only.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in parts]
    if unknown:
        raise SystemExit(
            f"--only: unknown part(s) {unknown}. Known parts: {list(parts)}"
        )
    return {name: parts[name] for name in wanted}


_REVISION_FILE = Path(__file__).resolve().parents[1] / "REVISION"


def _revision_stamp() -> dict[str, str]:
    """Commit and dirty state recorded at deploy time, for hosts without .git.

    The graceful degradation below did exactly what it was designed to do and
    that turned out to be the problem: every cluster run has an EMPTY git_commit,
    because the code is rsynced without .git. The runs that most need a revision
    are the ones that never had one.

    So the deploy step writes a REVISION file next to the code, and it is read
    here when git itself is unavailable. Two lines: the commit, then the
    porcelain status at deploy time (empty if the tree was clean). NGBP_GIT_COMMIT
    and NGBP_GIT_DIRTY override both, for a scheduler that would rather pass them
    in the environment.
    """
    env = os.environ.get("NGBP_GIT_COMMIT", "").strip()
    if env:
        return {"commit": env, "dirty": os.environ.get("NGBP_GIT_DIRTY", "").strip(),
                "source": "env"}
    try:
        lines = _REVISION_FILE.read_text().splitlines()
        return {"commit": lines[0].strip() if lines else "",
                "dirty": "\n".join(lines[1:]).strip(),
                "source": "REVISION file"}
    except Exception:
        return {"commit": "", "dirty": "", "source": "unavailable"}


def _git(*args: str) -> str:
    """A git query that degrades to "" rather than killing an experiment.

    Runs on compute nodes and from exported tarballs where .git may be absent,
    so every failure mode -- no repo, no git binary, timeout -- returns the empty
    string. Callers that need the commit should go through `provenance`, which
    falls back to `_revision_stamp` rather than recording nothing.
    """
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def provenance() -> dict[str, str]:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    stamp = {"source": "git"}
    if not commit:
        stamp = _revision_stamp()
        commit, dirty = stamp["commit"], stamp["dirty"]
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Non-empty means the tree carried uncommitted changes when this ran, so
        # git_commit alone does not reproduce the result.
        "git_dirty": dirty,
        # Which of git / the REVISION file / the environment supplied the commit,
        # so "" is distinguishable from "recorded, from a deploy stamp".
        "revision_source": stamp["source"],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cpu_count": str(os.cpu_count()),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
