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
DOCS=("../../overleaf/paper:9:paper" "../../overleaf/workshop:4:workshop")
PAPER=../../overleaf/paper/main.tex
APPENDIX=../../overleaf/paper/appendix.tex
BUILD=${BUILD_DIR:-/tmp/paper-check}
FAIL=0

pass() { printf "  [PASS] %s\n" "$1"; }
fail() { printf "  [FAIL] %s\n" "$1"; FAIL=1; }

echo "Paper gate"
echo

# Every check below greps files. If a path is wrong, grep finds nothing and the
# check passes -- which is exactly what happened when the documents moved to
# overleaf/ and this script still pointed at the old tree. So: exist first.
for required in "$PAPER" "$APPENDIX" ../../overleaf/workshop/main.tex; do
    [ -f "$required" ] || { fail "missing input: $required"; }
done
if [ "$FAIL" -ne 0 ]; then
    echo
    echo "Gate cannot run -- the documents are not where this script expects."
    exit 1
fi

# 1. No code-file references in the paper. Pointers to the implementation
#    belong in the compendium; the paper is the product, not the workshop.
n=$(cat "$PAPER" ../../overleaf/workshop/main.tex | grep -c 'codefile\|coderef' || true)
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
n=$(grep -c 'needsdata' "$PAPER" "$APPENDIX" ../../overleaf/workshop/main.tex ../../overleaf/shared/sections/*.tex 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
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
        dir="${spec%%:*}"; rest="${spec#*:}"; limit="${rest%%:*}"; label="${rest#*:}"
        base=main
        # Compiled from inside its own folder, because the roots resolve
        # `../shared/...` relative to themselves -- the same way Overleaf does it.
        #
        # -k keeps the .aux, which is where the LastMainPage page number is read
        # from below. Without it tectonic deletes the intermediates and the check
        # silently falls back to the text heuristic it is meant to replace.
        if (cd "$dir" && tectonic -X compile -k main.tex --outdir "$BUILD" >/dev/null 2>&1); then
            # \label{LastMainPage} sits immediately before \bibliographystyle, so
            # LaTeX itself reports which page the main content ends on. This used
            # to be found by hunting extracted text for "References" next to a
            # "[1]" -- a heuristic that a style change or a stray match breaks,
            # and one that fails silently in the permissive direction.
            body=$(sed -n 's/.*\\newlabel{LastMainPage}{{[^}]*}{\([0-9]*\)}.*/\1/p' \
                   "$BUILD/$base.aux" 2>/dev/null | head -1)
            if [ -z "$body" ]; then
                # Fall back to the old heuristic rather than skipping the check.
                body=$(pdftotext -layout "$BUILD/$base.pdf" - 2>/dev/null | awk '
                    BEGIN { RS="\f"; p=0 }
                    /References/ && (/\[1\]/ || /\[2\]/) { print NR-1; p=1; exit }
                    END { if (!p) print -1 }')
                echo "         (no LastMainPage label in $label; used the text heuristic)"
            fi
            if [ "$body" -lt 0 ]; then
                fail "$label: could not locate the end of main content"
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

# 4b. Every section file must be reached by at least one production document.
#
#     The rule used to be "reached by BOTH", which was right while the workshop
#     was a compression of the paper. It is not right now that the workshop is a
#     strict subset -- the efficiency table and Fisher's identity are the paper's
#     alone -- and a gate that demands the impossible is a gate people switch off.
#     What still matters is that no section is orphaned: an unreached file is a
#     copy nobody compiles, so it drifts from whatever superseded it and is then
#     available to be pasted back in. That is exactly how the stale ring model
#     (missing its normaliser) outlived the corrected one in the compendium.
for f in ../../overleaf/shared/sections/*.tex; do
    name=$(basename "$f" .tex)
    inmain=$(grep -c "input{../shared/sections/$name}" "$PAPER" || true)
    inws=$(grep -c "input{../shared/sections/$name}" ../../overleaf/workshop/main.tex || true)
    if [ "$inmain" -ge 1 ] || [ "$inws" -ge 1 ]; then
        pass "sections/$name reached (main:$inmain workshop:$inws)"
    else
        fail "sections/$name is orphaned -- reached by neither document"
    fi
done

# 4c. Efficiency figures in the prose must come from the generator's macros.
#
#     The abstract said "between 8 and 14" for three weeks after the generator
#     started emitting 7.3-15.7, because one number was typed in main.tex and the
#     other computed in tools/. The first version of this check compared the two
#     numerically and needed a tolerance to pass -- which is the same bug one level
#     up, since the tolerance gets widened until it goes green. So the rule is
#     structural instead: the numbers live in one generated file, the prose cites
#     macros, and a typed decimal near the word "ratio" is the failure.
MACROS=../../overleaf/shared/sections/efficiency-numbers.tex
if [ ! -f "$MACROS" ]; then
    fail "$MACROS missing -- run tools/make_tab_efficiency.py"
elif ! grep -q 'input{../shared/sections/efficiency-numbers}' "$PAPER"; then
    fail "$PAPER does not \\input sections/efficiency-numbers"
else
    pass "efficiency figures come from generated macros"
fi

# A typed ratio range anywhere in the production documents. Matches "between $8$
# and $14$" and "$8$--$14$"; the macro forms (\ratiolo, \ratioloword) contain no
# digits and so cannot trip it.
# The thesis is included here even though it is not otherwise gated. It is the
# document that drifted: the conclusion carried 7.3/12.6/15.7 for a day after the
# certified run emitted 7.0/12.5/15.5, because it was the one root not wired to
# the generated macros. Length and placeholders stay ungated for the thesis --
# it is allowed to be long and provisional -- but a stale headline number is not
# a matter of taste.
typed=$(grep -nE 'between \$[0-9]+(\.[0-9])?\$ and \$[0-9]+(\.[0-9])?\$ ?(times|\\times)|\$[0-9]+(\.[0-9])?\$--\$[0-9]+(\.[0-9])?\$ ?\\times' \
        "$PAPER" ../../overleaf/workshop/main.tex \
        ../../overleaf/thesis/chapters/ch12-conclusions.tex 2>/dev/null || true)
if [ -z "$typed" ]; then
    pass "no hand-typed efficiency ratio ranges"
else
    fail "hand-typed ratio range -- use the generated macros:"
    echo "$typed" | sed 's/^/         /'
fi

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
