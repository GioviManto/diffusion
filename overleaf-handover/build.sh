#!/usr/bin/env bash
# Compile all four documents and report page counts.
#
# `-k` keeps the intermediates so bibtex resolves on the second pass; without
# it every \cite renders as [?] and the build still exits zero, which is the
# kind of green-but-wrong this script exists to avoid. So citations are checked
# explicitly below rather than trusted to the exit code.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

fail=0
for f in workshop paper compendium thesis; do
    printf '%-12s ' "$f"
    if ! tectonic -k -o . "$f.tex" > "/tmp/handover_$f.log" 2>&1; then
        echo "BUILD FAILED"
        grep -E '^error|^!' "/tmp/handover_$f.log" | head -5 | sed 's/^/             /'
        fail=1
        continue
    fi
    undef=$(grep -icE 'undefined (reference|citation)|multiply.defined' \
            "/tmp/handover_$f.log" | tr -d ' ')
    # pdfinfo, not `mdls`: mdls reads Spotlight's index, which has not caught up
    # with a PDF written seconds ago and answers "(null)".
    pages=$(pdfinfo "$f.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
    pages=${pages:-?}
    printf 'ok  %4s pp  ' "$pages"
    if [ "$undef" != "0" ]; then
        echo "$undef undefined reference/citation warning(s)"
        fail=1
    else
        echo "no undefined refs or citations"
    fi
done

# The red PENDING markers are the reason a document is not sendable. Attribute
# them PER ROOT by following \input transitively: appendix.tex is reached by
# paper.tex and not by workshop.tex, so a flat file list would report the
# workshop as unsendable when it is fine -- and a warning that names innocent
# documents is one people learn to skip.
#
# `\needsdata` uses only: notation.tex defines the macro with \providecommand
# and would otherwise be reported forever.
reach() {  # transitive \input closure of a root, as a file list
    local seen="" queue="$1" f t
    while [ -n "$queue" ]; do
        f="${queue%% *}"; queue="${queue#"$f"}"; queue="${queue# }"
        case " $seen " in *" $f "*) continue;; esac
        [ -f "$f" ] || continue
        seen="$seen $f"
        for t in $(grep -oE '\\input\{[^}]+\}' "$f" 2>/dev/null \
                   | sed 's/\\input{//; s/}//'); do
            case "$t" in *.tex) queue="$queue $t";; *) queue="$queue $t.tex";; esac
        done
    done
    echo "$seen"
}

echo
for f in workshop paper compendium thesis; do
    files=$(reach "$f.tex")
    pend=$(grep -n '\\needsdata{' $files 2>/dev/null \
           | grep -v 'providecommand' | grep -v '^\S*:[0-9]*:%')
    if [ -n "$pend" ]; then
        echo "$f: NOT sendable -- unfilled \\needsdata"
        echo "$pend" | sed 's/^/    /'
    fi
done

exit $fail
