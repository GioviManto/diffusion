#!/usr/bin/env bash
# Build self-contained, Overleaf-ready folders (and zips) for each document.
#
# Why this exists. The working layout keeps one copy of the figures, the
# notation and the generated number files in shared/, and each document reaches
# them with ../shared/... That is right for the repository and wrong for
# Overleaf, for two separate reasons:
#
#   1. Overleaf's bibtex will not read through `..`. It finds no .bib at all,
#      drops every citation, and still exits zero -- so the document compiles
#      and looks fine until you scroll to the references and find them gone.
#
#   2. Whether a relative \input resolves against the project root or against
#      the main file's own directory is a property of the compiler invocation,
#      not of the document. Depending on it is a bet.
#
# So this flattens: every file a document needs is copied beside it and every
# ../shared/ prefix is rewritten away. The result has no parent-directory
# reference anywhere and compiles the same wherever it is unpacked. Upload
# upload/<doc>.zip to Overleaf, or drag the upload/<doc>/ folder in.
#
# These are build products. They are gitignored; regenerate rather than edit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DOCS=(thesis paper workshop compendium)
OUT=upload
rm -rf "$OUT"; mkdir -p "$OUT"

for d in "${DOCS[@]}"; do
    b="$OUT/$d"
    mkdir -p "$b"

    # The document's own sources. .sty matters as much as .tex here: the paper
    # and the workshop note carry neurips_2023.sty locally, and a bundle
    # without it fails on the first \usepackage rather than at the references.
    cp -R "$d"/*.tex "$b"/
    for ext in sty cls bst clo; do
        cp "$d"/*."$ext" "$b"/ 2>/dev/null || true
    done
    [ -d "$d/chapters" ] && cp -R "$d/chapters" "$b"/

    # everything it reaches through ../shared/, brought alongside
    cp -R shared/figures "$b"/figures
    cp -R shared/sections "$b"/sections
    cp    shared/notation.tex   "$b"/notation.tex
    cp    shared/references.bib "$b"/references.bib

    # exploratory/ is not referenced by any document; it only bloats the upload
    rm -rf "$b/figures/exploratory"

    # the per-folder bib copy the repo carries for the old scheme, if present
    rm -f "$b/references.bib.orig"

    # rewrite every parent-directory reference. Order matters: the longest
    # prefixes first, so ../shared/figures/ is not half-rewritten by the
    # ../shared/ rule behind it.
    find "$b" -name '*.tex' -print0 | xargs -0 perl -pi -e '
        s{\.\./shared/figures/}{figures/}g;
        s{\.\./shared/sections/}{sections/}g;
        s{\.\./shared/notation}{notation}g;
        s{\.\./shared/references}{references}g;
        s{\.\./shared/}{}g;
    '

    # Nothing may still point outside the bundle. The lookbehind matters:
    # the paper's appendix typesets elided paths like exp_01_.../grid.csv, and
    # a plain ../ search reports every one of them as a parent reference.
    if grep -rnP '(?<!\.)\.\./' "$b" --include='*.tex' >/dev/null 2>&1; then
        echo "  $d: REFUSING -- a parent-directory reference survived:" >&2
        grep -rnP '(?<!\.)\.\./' "$b" --include='*.tex' | sed 's/^/      /' >&2
        exit 1
    fi

    # The real test is that it builds standing alone, in a directory that has
    # no shared/ beside it at all. A bundle that only compiles next to its
    # source tree has not been flattened, it has been copied.
    if command -v tectonic >/dev/null 2>&1; then
        log=$(mktemp)
        if ! ( cd "$b" && tectonic -k --print main.tex ) >"$log" 2>&1; then
            echo "  $d: REFUSING -- bundle does not compile standalone" >&2
            grep -E '^error|^!' "$log" | head -5 | sed 's/^/      /' >&2
            exit 1
        fi
        last=$(awk '/^note: (Re)?[Rr]unning TeX/{buf=""} {buf=buf $0 "\n"}
                    END{printf "%s", buf}' "$log")
        # `|| true` is load-bearing under `set -e`: grep -c exits 1 when the
        # count is zero, which is the case we want.
        bad=$(printf '%s' "$last" | grep -icE 'undefined (reference|citation)' || true)
        if [ "$bad" != "0" ]; then
            echo "  $d: REFUSING -- $bad undefined reference/citation in the bundle" >&2
            exit 1
        fi
        pages=$(pdfinfo "$b/main.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
        rm -f "$b"/main.pdf "$b"/main.aux "$b"/main.bbl "$b"/main.blg \
              "$b"/main.log "$b"/main.out "$b"/main.toc "$b"/main.lof \
              "$b"/main.lot "$b"/main.run.xml "$b"/main.bcf 2>/dev/null || true
    else
        pages="?"
    fi

    ( cd "$OUT" && zip -qr "$d.zip" "$d" )
    printf '  %-12s %-24s %5s pp  %s\n' "$d" "$OUT/$d.zip" "$pages" "$(du -sh "$b" | cut -f1 | tr -d ' ')"
done

# One archive carrying all four, as sibling self-contained folders. This is the
# answer to "can I just upload the whole thing": not overleaf/ as it stands,
# because of the ../shared/ bibliography, but this, which is the same four
# bundles already proven to compile standing alone. Overleaf's main-document
# selector then switches between them. The figures are duplicated four times,
# which costs a few megabytes and buys independence from how Overleaf resolves
# a relative path.
( cd "$OUT" && zip -qr all.zip thesis paper workshop compendium )

echo
echo "  all four        $OUT/all.zip            $(du -sh "$OUT/all.zip" | cut -f1 | tr -d ' ')"
echo
echo "Overleaf: New Project -> Upload Project."
echo "  one document   -> upload/<doc>.zip,  main.tex is already the main file"
echo "  all four       -> upload/all.zip,    then Menu -> Main document to switch"
