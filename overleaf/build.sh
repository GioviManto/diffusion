#!/usr/bin/env bash
# Compile all four documents and report page counts.
#
# Each document lives in its own folder and is compiled from inside it, so the
# `../shared/...` paths in the roots resolve the same way here as they do in
# Overleaf when you set that file as the main document.
#
# `-k` keeps the intermediates: bibtex needs a second pass to resolve, and the
# .aux is where the main-content page count is read from. Without it every
# \cite renders as [?] and the build still exits zero -- the kind of
# green-but-wrong this script exists to catch, which is why citations and page
# limits are checked explicitly rather than trusted to the exit code.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# name:page limit for main content (0 = no limit)
DOCS=("workshop:4" "paper:9" "compendium:0" "thesis:0")
fail=0

for spec in "${DOCS[@]}"; do
    d="${spec%%:*}"; limit="${spec#*:}"
    printf '%-12s ' "$d"
    log="/tmp/overleaf_$d.log"
    if ! (cd "$d" && tectonic -k main.tex) > "$log" 2>&1; then
        echo "BUILD FAILED"
        grep -E '^error|^!' "$log" | head -5 | sed 's/^/             /'
        fail=1
        continue
    fi

    undef=$(grep -icE 'undefined (reference|citation)|multiply.defined' "$log" | tr -d ' ')
    # pdfinfo, not `mdls`: mdls reads Spotlight's index, which has not caught up
    # with a PDF written seconds ago and answers "(null)".
    pages=$(pdfinfo "$d/main.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')

    # Main content ends at \label{LastMainPage}, which sits just before the
    # bibliography. Reading it from the .aux beats hunting extracted text for
    # "References" next to a "[1]" -- that heuristic undercounts by a page
    # whenever the bibliography starts on the page the text ends on.
    body=$(sed -n 's/.*\\newlabel{LastMainPage}{{[^}]*}{\([0-9]*\)}.*/\1/p' \
           "$d/main.aux" 2>/dev/null | head -1)

    printf 'ok  %4s pp total' "${pages:-?}"
    [ -n "$body" ] && printf ', %s body' "$body"

    if [ "$undef" != "0" ]; then
        printf '  -- %s undefined reference/citation warning(s)\n' "$undef"
        fail=1
    elif [ "$limit" != "0" ] && [ -n "$body" ] && [ "$body" -gt "$limit" ]; then
        printf '  -- OVER LIMIT (%s > %s)\n' "$body" "$limit"
        fail=1
    else
        printf '  -- clean\n'
    fi
done

# Unfilled \needsdata markers, attributed per root by following \input
# transitively: appendix.tex is reached by the paper and not by the workshop, so
# a flat file list would report the workshop as unsendable when it is fine, and a
# warning that names innocent documents is one people learn to skip.
reach() {  # transitive \input closure of a root, as a file list
    local seen="" queue="$1" f t
    while [ -n "$queue" ]; do
        f="${queue%% *}"; queue="${queue#"$f"}"; queue="${queue# }"
        case " $seen " in *" $f "*) continue;; esac
        [ -f "$f" ] || continue
        seen="$seen $f"
        for t in $(grep -oE '\\input\{[^}]+\}' "$f" 2>/dev/null | sed 's/\\input{//; s/}//'); do
            case "$t" in
                /*) ;;
                *) t="$(dirname "$f")/$t" ;;
            esac
            case "$t" in *.tex) queue="$queue $t";; *) queue="$queue $t.tex";; esac
        done
    done
    echo "$seen"
}

echo
for spec in "${DOCS[@]}"; do
    d="${spec%%:*}"
    # notation.tex DEFINES \needsdata with \providecommand; skip definitions.
    pend=$(grep -n '\\needsdata{' $(reach "$d/main.tex") 2>/dev/null \
           | grep -v 'providecommand' | grep -v '^\S*:[0-9]*:%')
    if [ -n "$pend" ]; then
        echo "$d: NOT sendable -- unfilled \\needsdata"
        echo "$pend" | sed 's/^/    /'
        fail=1
    fi
done

exit $fail
