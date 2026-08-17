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

# The red PENDING markers are the reason two of these are not sendable. Report
# them here so nobody has to remember to look. Uses only: notation.tex defines
# the macro with \providecommand and would otherwise be reported forever,
# training everyone to ignore the line.
pend=$(grep -n '\\needsdata{' ./*.tex sections/*.tex 2>/dev/null \
       | grep -v 'providecommand' | grep -v '^\S*:[0-9]*:%')
if [ -n "$pend" ]; then
    echo
    echo "unfilled \\needsdata (these documents are not sendable):"
    echo "$pend" | sed 's/^/  /'
fi

exit $fail
