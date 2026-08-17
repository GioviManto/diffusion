#!/usr/bin/env bash
# Gate the paper against the rules the rewrite was done under.
#
# These are the checks from the rebuild plan, in one place, so "is the paper
# ready" has a mechanical answer rather than a judgement call. Run from the
# repository root:
#
#     tools/check_paper.sh
#
# Exits non-zero if any check fails. Intended for CI and for the final build.

set -uo pipefail
cd "$(dirname "$0")/.."

# Both production documents are gated. The compendium is the development
# environment and is deliberately NOT gated -- it is where unfinished work is
# allowed to live.
DOCS=("paper/main.tex:9:paper" "paper/workshop.tex:4:workshop")
PAPER=paper/main.tex
APPENDIX=paper/appendix.tex
BUILD=${BUILD_DIR:-/tmp/paper-check}
FAIL=0

pass() { printf "  [PASS] %s\n" "$1"; }
fail() { printf "  [FAIL] %s\n" "$1"; FAIL=1; }

echo "Paper gate"
echo

# 1. No code-file references in the paper. Pointers to the implementation
#    belong in the compendium; the paper is the product, not the workshop.
n=$(cat "$PAPER" paper/workshop.tex | grep -c 'codefile\|coderef' || true)
[ "$n" -eq 0 ] && pass "no \\codefile/\\coderef in either document" \
                || fail "$n \\codefile/\\coderef occurrence(s) in the production documents"

# 2. No self-correction language. Superseded analyses live in the compendium's
#    corrections chapter; a reader of the paper should meet claims, not
#    retractions of claims.
if grep -inE 'we withdraw|withdrawn|an earlier version|superseded|no longer rely|that reading is' "$PAPER" >/dev/null; then
    fail "correction/withdrawal language in main.tex:"
    grep -inE 'we withdraw|withdrawn|an earlier version|superseded|no longer rely|that reading is' "$PAPER" | sed 's/^/         /'
else
    pass "no correction/withdrawal language in main.tex"
fi

# 3. No unfilled data placeholders.
n=$(grep -c 'needsdata' "$PAPER" "$APPENDIX" paper/workshop.tex paper/sections/*.tex 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
if [ "$n" -eq 0 ]; then
    pass "no \\needsdata placeholders remain"
else
    fail "$n \\needsdata placeholder(s) still unfilled -- the frozen batch has not landed"
fi

# 4. Body length, per document. NeurIPS 2026 is 9 content pages; the workshop
#    target is 4. References and appendices are unlimited for both.
if command -v tectonic >/dev/null && command -v pdftotext >/dev/null; then
    mkdir -p "$BUILD"
    for spec in "${DOCS[@]}"; do
        src="${spec%%:*}"; rest="${spec#*:}"; limit="${rest%%:*}"; label="${rest#*:}"
        base=$(basename "$src" .tex)
        if (cd paper && tectonic -X compile "$base.tex" --outdir "$BUILD" >/dev/null 2>&1); then
            body=$(pdftotext -layout "$BUILD/$base.pdf" - 2>/dev/null | awk '
                BEGIN { RS="\f"; p=0 }
                /References/ && (/\[1\]/ || /\[2\]/) { print NR-1; p=1; exit }
                END { if (!p) print -1 }')
            if [ "$body" -lt 0 ]; then
                fail "$label: could not locate the bibliography"
            elif [ "$body" -le "$limit" ]; then
                pass "$label body is $body pages (limit $limit)"
            else
                fail "$label body is $body pages (limit $limit)"
            fi
        else
            fail "$label does not build"
        fi
    done
else
    echo "  [SKIP] body length -- needs tectonic and pdftotext"
fi

# 4b. The shared sections must actually be shared. A block that stops being
#     \input by both documents has silently become two copies, which is the
#     drift this structure exists to prevent.
for f in paper/sections/*.tex; do
    name=$(basename "$f" .tex)
    [ "$name" = "workshop-appendix" ] && continue
    inmain=$(grep -c "input{sections/$name}" paper/main.tex || true)
    inws=$(grep -c "input{sections/$name}" paper/workshop.tex || true)
    if [ "$inmain" -ge 1 ] && [ "$inws" -ge 1 ]; then
        pass "sections/$name shared by both documents"
    else
        fail "sections/$name is in main:$inmain workshop:$inws -- no longer shared"
    fi
done

# 5b. Replicate counts, checked block-aware.
#
#     Delegated to a Python checker because a line-based grep cannot tell a
#     `quick` smoke dict from a `full` one, and missed a real gap doing exactly
#     that (exp_27 at eight seeds). See tools/check_replicates.py for the four
#     gaps that motivated it.
if PY=$(command -v ./.venv/bin/python || command -v python3); then
    if "$PY" tools/check_replicates.py; then :; else FAIL=1; fi
else
    echo "  [SKIP] replicate counts -- no python found"
fi

# 5. Every experiment behind the paper imports the frozen config rather than
#    setting rho / grid / iteration counts locally.
stray=$(grep -ln 'RHO_TRUE\s*=\|rho_true\s*=\s*0\.8' experiments/exp_*.py 2>/dev/null \
        | xargs -r grep -L 'frozen_config' || true)
if [ -z "$stray" ]; then
    pass "no experiment sets rho outside the frozen config"
else
    fail "experiment(s) setting rho locally without importing frozen_config:"
    echo "$stray" | sed 's/^/         /'
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "All checks passed."
else
    echo "Gate failed."
fi
exit "$FAIL"
