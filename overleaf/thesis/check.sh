#!/usr/bin/env bash
# Build the thesis and report every defect class, refusing to look at a stale PDF.
#
# WHY THE EXIT CODE IS CHECKED FIRST. tectonic leaves the previous main.pdf in
# place when a run fails. A check that compiles with output suppressed and then
# inspects main.pdf will happily report "0 unresolved references, 144 pages" from
# a PDF built before the error -- which is exactly what happened on 4 September
# 2026 while adding Appendix B: a `Double subscript` error halted the build for
# three consecutive "clean" reports. Build first, check the status, and only then
# read the artefact.
#
#   ./check.sh          build and report
#   ./check.sh --quiet  report only the summary line
#
# Exits non-zero if the build fails or any defect class is non-empty.

set -uo pipefail
cd "$(dirname "$0")"

QUIET=0
[[ ${1:-} == "--quiet" ]] && QUIET=1
say() { [[ $QUIET -eq 1 ]] || echo "$@"; }

log=$(mktemp)
if ! tectonic -X compile main.tex --keep-intermediates --keep-logs --reruns 4 \
        >"$log" 2>&1; then
    echo "BUILD FAILED -- main.pdf is stale and was not inspected:" >&2
    grep -iE '^error|^!' "$log" | head -10 | sed 's/^/    /' >&2
    exit 1
fi
say "build ok"

python3 - "$QUIET" <<'PY'
import re, subprocess, sys, pathlib
quiet = sys.argv[1] == "1"
log = pathlib.Path("main.log").read_text(errors="ignore")
last = log.rsplit("Running TeX", 1)[-1]        # final pass only
pdf = subprocess.run(["pdftotext", "main.pdf", "-"],
                     capture_output=True, text=True).stdout
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True).stdout

overfull = [float(x) for x in
            re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', last)]
defects = {
    "undefined references": len(set(re.findall(r"Reference `([^']+)' on page", last))),
    "undefined citations":  len(set(re.findall(r"Citation `([^']+)' on page", last))),
    "multiply-defined labels": len(set(re.findall(r"Label `([^']+)' multiply", last))),
    "'??' printed in the PDF": pdf.count("??"),
    "missing glyphs": len(re.findall(r'Missing character', last)),
    # A few points over is invisible; 10pt is a word in the margin.
    "overfull hboxes > 10pt": sum(1 for v in overfull if v > 10),
}
pages = info.split("Pages:")[1].split()[0]
figs = len(set(re.findall(r'Figure (\d+\.\d+):', pdf)))
tabs = len(set(re.findall(r'Table (\d+\.\d+):', pdf)))
cites = len(re.findall(r'.bibitem', pathlib.Path("main.bbl").read_text()))

if not quiet:
    for k, v in defects.items():
        print(f"  {k:<26}{v}")
    print(f"  {'overfull (any size)':<26}{len(overfull)} "
          f"(worst {max(overfull, default=0):.1f}pt)")
    print(f"  {'pages':<26}{pages}")
    print(f"  {'figures / tables':<26}{figs} / {tabs}")
    print(f"  {'citations':<26}{cites}")

bad = {k: v for k, v in defects.items() if v}
if bad:
    print("DEFECTS: " + ", ".join(f"{k}={v}" for k, v in bad.items()))
    sys.exit(1)
print(f"clean: {pages} pp, {figs} figures, {tabs} tables, {cites} citations")
PY
