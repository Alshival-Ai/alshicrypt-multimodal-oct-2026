#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Short local study: 10-epoch screening, 20-epoch repeated trials.
# Resumable, but never change experiment code while a campaign is running.
.venv/bin/python -m pytest -q
for stage in screen extend test diagnose; do
    .venv/bin/python -m alshicrypt.cli "$stage" --extended-steps 2000 "$@"
done
.venv/bin/python scripts/research_report.py "$@"
if [[ -x .tools/tectonic ]]; then
    .tools/tectonic research/main.tex --outdir research
elif command -v pdflatex >/dev/null 2>&1; then
    (cd research && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex)
fi
