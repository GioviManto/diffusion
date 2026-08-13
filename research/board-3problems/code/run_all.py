"""Run every audit driver and print one verdict. Exits non-zero on any failure.

    python code/run_all.py            (from the board-3problems directory)
"""

import pathlib
import subprocess
import sys

DRIVERS = [
    ("verify_ring_walk.py", "gauge reduction + closed-form joint score"),
    ("audit_three_problems.py", "the three board problems at T = 2"),
    ("verify_scaling.py", "scaling claims H1, H2 (both rejected)"),
]

here = pathlib.Path(__file__).resolve().parent
rows, failed = [], []

for script, label in DRIVERS:
    print(f"\n{'=' * 72}\n>>> {script} -- {label}\n{'=' * 72}")
    r = subprocess.run([sys.executable, str(here / script)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    n_pass = r.stdout.count("[PASS]")
    n_fail = r.stdout.count("[FAIL]")
    rows.append((script, n_pass, n_fail, r.returncode))
    if r.returncode != 0:
        failed.append(script)

print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
tot_p = tot_f = 0
for script, p, f, rc in rows:
    tot_p += p
    tot_f += f
    print(f"  {script:<28s} {p:3d} passed  {f:3d} failed  "
          f"{'OK' if rc == 0 else 'FAILED'}")
print(f"  {'TOTAL':<28s} {tot_p:3d} passed  {tot_f:3d} failed")

if failed:
    print(f"\nFAILED: {failed}")
    sys.exit(1)
print("\nALL DRIVERS PASSED")
