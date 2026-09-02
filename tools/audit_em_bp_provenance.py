#!/usr/bin/env python3
"""Audit the EM-BP result package for the failure modes that actually occurred.

Written against real defects found on this branch, not a generic checklist. Each
check exists because something it would have caught was found by hand:

  manifest-vs-data     `params_clean_vs_noised.json` recorded `n_rep_clean=12`
                       while the committed CSV held 3 replicates. The manifest
                       described a run whose outputs were never committed.

  truth-consistency    the paper asserted "every experiment initialises rho at
                       0.3 against a truth of 0.85" while exp_06/07/08 use 0.8.
                       Different experiments may legitimately differ; what is not
                       allowed is a blanket claim, or a manifest disagreeing with
                       the source constant that produced it.

  stale-figures        `[0.8497, 0.8507]` appeared in the paper, the compendium
                       and two summary documents, and reproduced from no
                       committed output; the current exp_18 trace gives
                       [0.8517, 0.8520].

  schema               a part emitted rows with regime-dependent keys, so
                       `csv.DictWriter` raised on the first row of the second
                       regime -- after the compute had been done.

Exit code is 1 if any publication-blocking inconsistency is found, 0 otherwise,
so this can gate a commit.

Usage:
    python3 tools/audit_em_bp_provenance.py [--root research/nongaussian-bp]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from pathlib import Path

# Truth constants each experiment declares in its own source. Audited against
# both the manifests and the prose. Kept explicit rather than parsed so that a
# silent edit to a source constant shows up here as a mismatch.
# Which experiments this audit knows how to check. The VALUE each should carry
# is not written here on purpose.
#
# It used to be, as a literal per experiment, and it went stale the moment the
# project moved rho into experiments/frozen_config.py: six experiments changed
# from a local 0.8 to FROZEN.rho = 0.85, the recorded data followed, and this
# table did not. The audit then reported 174 blocking issues on correct data
# for as long as nobody read past the first eight lines of its output -- which
# is worse than not running, because a check that always fails is indistinguish-
# able from one that never passes and gets ignored the same way.
#
# So the expected value is resolved from the experiment's own source, which
# resolves in turn to frozen_config.py wherever the source says FROZEN.rho.
# Same rule as the documents: derive it, do not retype it.
AUDITED_EXPERIMENTS = (
    "exp_02", "exp_03", "exp_06", "exp_07", "exp_08", "exp_16", "exp_18", "exp_22",
)

# Phrases that assert a single global truth. Any of these is a blanket claim and
# is false while the repository uses two.
BLANKET_TRUTH_PATTERNS = [
    r"[Ee]very experiment initialises \$?\\?corr\$? at \$?0\.3\$? against a truth",
    r"initialised at \$?0\.3\$? in every experiment against a truth",
]

# Numbers withdrawn because they reproduce from no committed output.
WITHDRAWN_VALUES = ["0.8497", "0.8507"]

# Documents whose job is to record corrections. They must be able to name a
# withdrawn value -- "we previously reported X, it is now Y" is the whole point
# of them -- so the withdrawn-value check skips them. Everywhere else, naming one
# means presenting it as a current result, which is the thing being prevented.
WITHDRAWAL_RECORDS = {
    "RESULTS_FOR_JEROME_AND_MARC.md",
    "FINAL_REVISION_REPORT.md",
    "REVISION_AUDIT.md",
    "AUDIT_NOTE.md",
    "CLAIMS_TO_UPDATE.md",
}


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.blocking: list[str] = []
        self.warnings: list[str] = []
        self.archived = 0

    def block(self, msg: str) -> None:
        self.blocking.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # ---------------------------------------------------------------- checks

    def _frozen_rho(self) -> float | None:
        """The single rho declared in experiments/frozen_config.py."""
        if getattr(self, "_frozen_rho_cache", "unset") != "unset":
            return self._frozen_rho_cache
        self._frozen_rho_cache = None
        cfg = self.root / "experiments" / "frozen_config.py"
        if cfg.exists():
            m = re.search(r"^\s*rho:\s*float\s*=\s*([0-9.]+)", cfg.read_text(), re.M)
            if m:
                self._frozen_rho_cache = float(m.group(1))
        return self._frozen_rho_cache

    def source_rho(self, exp: str) -> set[float]:
        """The rho values an experiment's own source declares.

        A source may state a literal, sweep a tuple, or -- the current
        convention -- defer to FROZEN.rho. All three resolve here, so the audit
        compares recorded data against what the code actually says rather than
        against a copy of it kept in this file.
        """
        matches = list((self.root / "experiments").glob(f"{exp}_*.py"))
        if not matches:
            self.warn(f"{exp}: no source file found")
            return set()
        text = matches[0].read_text()
        vals: set[float] = set()

        # deferred to the frozen config -- the convention every current
        # experiment follows
        if re.search(r"^\s*RHO(?:_TRUE)?\s*=\s*FROZEN\.rho\b", text, re.M):
            frozen = self._frozen_rho()
            if frozen is None:
                self.block(f"{exp}: defers to FROZEN.rho, which this audit cannot read")
            else:
                vals.add(frozen)

        found = re.findall(r"^RHO(?:_TRUE)?\s*=\s*([0-9.]+)", text, re.M)
        found += re.findall(r'^\s*"rho":\s*([0-9.]+)', text, re.M)
        found += re.findall(r"^RHOS\s*=\s*\(([^)]*)\)", text, re.M)
        for f in found:
            for piece in str(f).split(","):
                piece = piece.strip()
                if piece:
                    try:
                        vals.add(float(piece))
                    except ValueError:
                        pass
        if not vals:
            self.warn(f"{exp}: no rho constant found in {matches[0].name}")
        return vals

    def check_source_truths(self) -> None:
        """Every audited experiment declares a rho this audit can resolve."""
        for exp in AUDITED_EXPERIMENTS:
            self.source_rho(exp)

    def _superseded(self, manifest: pathlib.Path) -> bool:
        """True if this manifest sits in, or under, a directory marked SUPERSEDED."""
        d = manifest.parent
        while True:
            if (d / "SUPERSEDED").exists():
                return True
            if d == self.root or d.parent == d:
                return False
            d = d.parent

    def check_manifest_truths(self) -> None:
        """Every params*.json records a truth matching its experiment's source."""
        for man in sorted((self.root / "outputs").rglob("params*.json")):
            try:
                data = json.loads(man.read_text())
            except Exception as exc:
                self.block(f"{self._rel(man)}: unreadable ({exc})")
                continue
            rho = data.get("rho_true", data.get("rho"))
            if rho is None:
                continue
            exp = self._experiment_of(man)
            if exp is None or exp not in AUDITED_EXPERIMENTS:
                continue
            # A run recorded before rho moved into frozen_config.py is not a
            # defect; it is a fact about an archived measurement. The directory
            # says so in a SUPERSEDED file, which also names the frozen
            # replacement. Marking beats deleting: the record survives, and
            # nothing silently reads it thinking it is current.
            if self._superseded(man):
                self.archived += 1
                continue
            declared = self.source_rho(exp)
            if not declared:
                continue
            if not any(abs(float(rho) - d) <= 1e-12 for d in declared):
                self.block(
                    f"{self._rel(man)}: records rho={rho}, "
                    f"but {exp} source declares {sorted(declared)}"
                )

    def check_manifest_matches_data(self) -> None:
        """Replicate counts in a manifest match the CSV committed beside it.

        This is the `n_rep_clean=12` vs 3-replicate mismatch. A manifest that
        describes a larger run than the data it sits next to is worse than no
        manifest: it makes an under-powered result look adequately replicated.
        """
        pairs = [
            ("params_clean_vs_noised.json", "n_rep_clean",
             "clean_vs_noised_params.csv", "rep"),
            ("params_clean_raw_mle.json", "n_rep_raw",
             "clean_raw_mle.csv", "rep"),
        ]
        for man_name, key, csv_name, col in pairs:
            for man in sorted((self.root / "outputs").rglob(man_name)):
                csv_path = man.parent / csv_name
                if not csv_path.exists():
                    continue
                try:
                    declared = int(json.loads(man.read_text())[key])
                except (KeyError, ValueError, TypeError):
                    continue
                reps = {r[col] for r in self._rows(csv_path) if r.get(col) not in (None, "")}
                if not reps:
                    continue
                actual = len(reps)
                if actual != declared:
                    self.block(
                        f"{self._rel(csv_path)}: holds {actual} replicates, but "
                        f"{man_name} declares {key}={declared}"
                    )

    def check_csv_schemas(self) -> None:
        """Every row of a CSV carries every column -- no regime-dependent keys."""
        for path in sorted((self.root / "outputs").rglob("*.csv")):
            rows = self._rows(path)
            if not rows:
                continue
            header = set(rows[0].keys())
            for i, row in enumerate(rows[1:], start=2):
                if set(row.keys()) != header:
                    self.block(
                        f"{self._rel(path)}: row {i} has a different column set "
                        f"than the header"
                    )
                    break
                if any(v is None for v in row.values()):
                    self.block(f"{self._rel(path)}: row {i} is short of fields")
                    break

    def check_withdrawn_values(self) -> None:
        """Withdrawn numbers must not reappear in prose."""
        exts = ("*.tex", "*.md")
        skip = {".venv", ".git", "node_modules", ".claude", "outputs"}
        for ext in exts:
            for path in sorted(self.root.rglob(ext)):
                if set(path.parts) & skip or path.suffix == ".aux":
                    continue
                if path.name in WITHDRAWAL_RECORDS:
                    continue
                text = path.read_text(errors="ignore")
                for bad in WITHDRAWN_VALUES:
                    if bad in text:
                        self.block(
                            f"{self._rel(path)}: contains withdrawn value {bad} "
                            "(reproduces from no committed output)"
                        )

    def check_blanket_truth_claims(self) -> None:
        """No document may assert one global truth while two are in use."""
        for ext in ("*.tex", "*.md"):
            for path in sorted(self.root.rglob(ext)):
                if set(path.parts) & {".venv", ".git", ".claude"}:
                    continue
                text = path.read_text(errors="ignore")
                for pat in BLANKET_TRUTH_PATTERNS:
                    m = re.search(pat, text)
                    if m:
                        line = text[: m.start()].count("\n") + 1
                        self.block(
                            f"{self._rel(path)}:{line}: blanket truth claim "
                            f"({m.group(0)[:60]!r}); the repo uses both 0.8 and 0.85"
                        )

    def check_overleaf_in_sync(self) -> None:
        """The Overleaf copy must not drift from paper/ -- it has before."""
        paper, over = self.root / "paper", self.root / "overleaf"
        if not over.exists():
            return
        for name in ("main.tex", "appendix.tex", "notation.tex"):
            a, b = paper / name, over / name
            if a.exists() and b.exists() and a.read_bytes() != b.read_bytes():
                self.block(
                    f"overleaf/{name} differs from paper/{name}; "
                    "run overleaf/sync.sh"
                )

    # --------------------------------------------------------------- helpers

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    @staticmethod
    def _experiment_of(path: Path) -> str | None:
        m = re.search(r"(exp_\d+)", str(path))
        return m.group(1) if m else None

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        try:
            with path.open(newline="") as fh:
                return list(csv.DictReader(fh))
        except Exception:
            return []

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        for name in [n for n in dir(self) if n.startswith("check_")]:
            getattr(self, name)()

        if self.warnings:
            print("WARNINGS")
            for w in self.warnings:
                print(f"  ~ {w}")
            print()

        if self.blocking:
            print("BLOCKING")
            for b in self.blocking:
                print(f"  x {b}")
            print(f"\n{len(self.blocking)} blocking issue(s).")
            return 1

        if self.archived:
            print(f"({self.archived} manifest(s) skipped: directory marked SUPERSEDED.)")
        print("All provenance checks passed.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent.parent / "research/nongaussian-bp",
    )
    args = ap.parse_args()
    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2
    return Audit(args.root).run()


if __name__ == "__main__":
    raise SystemExit(main())
