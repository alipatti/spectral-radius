#!/bin/sh
# bundle the paper sources and figures into a tarball for arxiv.
# usage: scripts/make-arxiv-tarball.sh [output] (default: build/for-arxiv.tar.gz)
set -eu

# resolve the output against the caller's directory before moving to the repo root
tarball=${1:-build/for-arxiv.tar.gz}
case $tarball in
/*) ;;
*) tarball="$PWD/$tarball" ;;
esac

cd "$(dirname "$0")/.."

if [ ! -d figures ]; then
	echo "warning: figures/ not found, run the figure scripts first" >&2
fi

mkdir -p "$(dirname "$tarball")"

{
	find paper -type f \( \
		-name '*.tex' -o -name '*.cls' -o -name '*.sty' -o -name '*.bib' \
		\)
	if [ -d figures ]; then
		find figures -type f \( -name '*.pdf' -o -name '*.pgf' \)
	fi
} | sort | tar -czf "$tarball" -T -

echo "$tarball"
tar -tzf "$tarball"
