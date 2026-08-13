"""Merge sharded sweep CSVs and report completeness.

Run after the arrays finish:

    python merge_shards.py --root ~/bp-em-diffusion/outputs

For each experiment it writes ``<experiment>_all.csv`` next to the shards and prints a
completeness summary: how many cells were expected, how many landed, and how many failed.
An incomplete merge is reported loudly rather than silently producing a short file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def merge_one(exp_dir: Path) -> dict:
    name = exp_dir.name
    shards = sorted(exp_dir.glob(f"{name}_shard*.csv"))
    metas = sorted(exp_dir.glob(f"{name}_shard*.meta.json"))

    rows: list[dict] = []
    keys: list[str] = []
    for s in shards:
        with s.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
                for k in r:
                    if k not in keys:
                        keys.append(k)

    n_expected = 0
    n_failed_meta = 0
    elapsed = 0.0
    for m in metas:
        d = json.loads(m.read_text())
        n_expected += int(d.get("n_cells", 0))
        n_failed_meta += int(d.get("n_failed", 0))
        elapsed += float(d.get("elapsed_s", 0.0))

    n_err_rows = sum(1 for r in rows if r.get("error"))

    if rows:
        out = exp_dir / f"{name}_all.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    return {
        "experiment": name,
        "shards": len(shards),
        "metas": len(metas),
        "rows": len(rows),
        "expected": n_expected,
        "failed_rows": n_err_rows,
        "failed_meta": n_failed_meta,
        "cpu_seconds": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).expanduser()

    any_problem = False
    for exp_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        s = merge_one(exp_dir)
        complete = s["rows"] == s["expected"] and s["expected"] > 0
        flag = "OK " if complete and s["failed_rows"] == 0 else "!! "
        if not complete or s["failed_rows"]:
            any_problem = True
        print(
            f"{flag}{s['experiment']:<20s} shards={s['shards']:<4d} "
            f"rows={s['rows']:<6d} expected={s['expected']:<6d} "
            f"failed={s['failed_rows']:<4d} cpu={s['cpu_seconds'] / 3600:6.2f} h"
        )
    if any_problem:
        print("\nSome experiments are incomplete or contain failed cells; "
              "inspect the 'error' column and the shard logs before using the merged CSV.")


if __name__ == "__main__":
    main()
