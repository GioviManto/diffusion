#!/usr/bin/env bash
# Refresh the thesis's local copies of the shared assets.
#
# This folder is deliberately self-contained: figures/, sections/ and
# references.bib are copies, not symlinks and not ../shared paths, so the
# folder can be zipped and uploaded to Overleaf on its own. The cost of that
# is drift, and this script is what removes it.
#
#   ./sync-assets.sh          refresh, then report what changed
#   ./sync-assets.sh --check  report only, exit 1 if anything is stale
#
# Figures rebuilt by tools/make_thesis_figures.py are NOT overwritten: they are
# the thesis's own versions and shared/figures still holds the ones the paper,
# the workshop note and the compendium were built against.

set -euo pipefail
cd "$(dirname "$0")"

SHARED=../shared
CHECK=0
[[ ${1:-} == "--check" ]] && CHECK=1

# Figures the thesis draws for itself. Never copied from shared/.
LOCAL_FIGURES=(
  fig_forward_corruption.pdf
  fig_three_scores.pdf
  fig_collapse_time.pdf
  fig_speciation_cascade.pdf
  fig_em_diagnostics.pdf
  fig_nonmarkov.pdf
  fig_screening.pdf
)

is_local() {
  local f
  for f in "${LOCAL_FIGURES[@]}"; do [[ $f == "$1" ]] && return 0; done
  return 1
}

stale=0
report() { echo "  $1"; stale=$((stale + 1)); }

# ---- figures --------------------------------------------------------------
for path in figures/*.pdf; do
  name=$(basename "$path")
  is_local "$name" && continue
  src="$SHARED/figures/$name"
  if [[ ! -f $src ]]; then
    report "orphan (not in shared/): $name"
  elif ! cmp -s "$src" "$path"; then
    report "stale figure: $name"
    [[ $CHECK -eq 0 ]] && cp "$src" "$path"
  fi
done

# ---- generated number and table sections ----------------------------------
for path in sections/*.tex; do
  name=$(basename "$path")
  src="$SHARED/sections/$name"
  if [[ ! -f $src ]]; then
    report "orphan (not in shared/): sections/$name"
  elif ! cmp -s "$src" "$path"; then
    report "stale section: $name"
    [[ $CHECK -eq 0 ]] && cp "$src" "$path"
  fi
done

# ---- bibliography ---------------------------------------------------------
if ! cmp -s "$SHARED/references.bib" references.bib; then
  report "stale references.bib"
  [[ $CHECK -eq 0 ]] && cp "$SHARED/references.bib" references.bib
fi

if [[ $stale -eq 0 ]]; then
  echo "up to date with $SHARED"
elif [[ $CHECK -eq 1 ]]; then
  echo
  echo "$stale item(s) stale. Run ./sync-assets.sh to refresh."
  exit 1
else
  echo
  echo "refreshed $stale item(s). Rebuild before uploading."
fi
