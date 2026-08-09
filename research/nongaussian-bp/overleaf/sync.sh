#!/usr/bin/env bash
# Refresh overleaf/ from paper/, then verify it still builds standalone.
#
# overleaf/ is a copy, not a symlink, because Overleaf needs a flat self-contained project.
# A copy silently goes stale the moment paper/ is edited -- which happened once already, and
# a stale copy is worse than no copy because it looks fine. Run this after touching paper/,
# or with --check in a pre-commit hook to fail loudly instead.
set -euo pipefail
cd "$(dirname "$0")/.."          # research/nongaussian-bp
FILES=(main.tex appendix.tex notation.tex bibliography.bib neurips_2023.sty)

if [[ "${1:-}" == "--check" ]]; then
    stale=0
    for f in "${FILES[@]}"; do
        cmp -s "paper/$f" "overleaf/$f" || { echo "STALE: $f"; stale=1; }
    done
    for f in paper/figures/*.pdf; do
        cmp -s "$f" "overleaf/figures/$(basename "$f")" || {
            echo "STALE: figures/$(basename "$f")"; stale=1; }
    done
    [[ $stale -eq 0 ]] && echo "overleaf/ is in sync with paper/"
    exit $stale
fi

for f in "${FILES[@]}"; do cp "paper/$f" "overleaf/$f"; done
cp paper/figures/*.pdf overleaf/figures/

# Build from a copy outside the repo: catches anything still reaching into paper/ by path.
tmp=$(mktemp -d)
cp -R overleaf "$tmp/proj"
( cd "$tmp/proj" && tectonic main.tex --keep-logs >/dev/null 2>&1 )
bad=$(grep -ciE 'undefined|multiply defined' "$tmp/proj/main.log" || true)
pages=$(python3 -c "import sys;print(open(sys.argv[1],'rb').read().count(b'/Type /Page)'))" \
        "$tmp/proj/main.pdf" 2>/dev/null || echo '?')
rm -rf "$tmp"
echo "overleaf/ synced and builds standalone; undefined references: $bad"
[[ "$bad" == "0" ]] || { echo "REFUSING: fix the undefined references first"; exit 1; }
