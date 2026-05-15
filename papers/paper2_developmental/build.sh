#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode paper.tex
bibtex paper || true
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
echo "Build complete: paper.pdf"
