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
    if [ -d "$d/chapters" ]; then
        cp -R "$d/chapters" "$b"/
        # Any underscore-prefixed directory is a local archive: _superseded
        # drafts, _withdrawn appendices. Nothing \input's them, so they do not
        # break the build -- they just ship confusing dead material to Overleaf
        # and eat into its 180-file cap. Matched by prefix rather than by name
        # so the next archive folder is excluded without editing this script.
        find "$b/chapters" -maxdepth 1 -type d -name '_*' -exec rm -rf {} +
    fi

    # Only what this document actually references. Copying all of shared/ was
    # padding the uploads badly -- the workshop note reaches one figure and
    # shipped twenty, plus twenty .png copies of them that no \includegraphics
    # ever names, plus every generated section belonging to the other three
    # documents. Overleaf caps a project at 180 files, so the padding was not
    # free.
    mkdir -p "$b/figures" "$b/sections"
    # A document may carry its own figures/ and sections/ instead of reaching
    # into shared/ -- the thesis does, so that restyling its figures cannot
    # disturb the other three. Its own copy wins; shared/ is the fallback.
    python3 - "$b" shared "$d" <<'PY'
import re, shutil, sys, pathlib
bundle, shared, doc = (pathlib.Path(a) for a in sys.argv[1:4])
text = "\n".join(p.read_text() for p in bundle.rglob("*.tex"))

def take(cands, dest):
    for c in cands:
        if c.exists():
            shutil.copy2(c, dest / c.name)
            return True
    return False

missing = []
for name in set(re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)):
    stem = pathlib.Path(name).name
    if not stem.endswith(".pdf"):
        stem += ".pdf"
    if not take([doc / "figures" / stem, shared / "figures" / stem],
                bundle / "figures"):
        missing.append(stem)

for name in set(re.findall(r"\\input\{(?:\.\./shared/)?sections/([^}]+)\}", text)):
    f = pathlib.Path(name).with_suffix(".tex").name
    if not take([doc / "sections" / f, shared / "sections" / f],
                bundle / "sections"):
        missing.append("sections/" + f)

if missing:
    sys.exit("  missing asset(s), nowhere in %s/ or shared/: %s"
             % (doc, ", ".join(sorted(missing))))
PY
    cp shared/notation.tex   "$b"/notation.tex
    cp shared/references.bib "$b"/references.bib

    # READMEs describe the repository layout, which the bundle does not have
    find "$b" -name 'README.md' -delete

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

    # Nothing may still point outside the bundle.
    #
    # The lookbehind matters: the paper's appendix typesets elided paths like
    # exp_01_.../grid.csv, and a plain ../ search reports every one of them.
    # This is perl and not `grep -P` because it has to run under whichever grep
    # the script's shell resolves -- BSD grep has no -P, so the check errored
    # out, the `if` read false, and the guard silently passed everything for as
    # long as it existed. A guard that cannot fail is not a guard.
    stray=$(find "$b" -name '*.tex' -print0 \
            | xargs -0 perl -ne 'print "$ARGV:$.: $_" if m{(?<!\.)\.\./}')
    if [ -n "$stray" ]; then
        echo "  $d: REFUSING -- a parent-directory reference survived:" >&2
        sed 's/^/      /' >&2 <<<"$stray"
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

# Deliberately no combined archive. One Overleaf project per document is what
# gets shared and commented on separately, and a fifth zip containing copies of
# the other four is exactly the kind of near-duplicate that gets uploaded by
# mistake. Each document is its own project.

echo
echo "Overleaf: New Project -> Upload Project -> upload/<doc>.zip"
echo "main.tex is already the main document in each; nothing to configure."
