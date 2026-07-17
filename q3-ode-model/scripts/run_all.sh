#!/usr/bin/env bash
# Run the full melanoma ODE pipeline (phases 1-6) end to end.
# Usage:  bash scripts/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HERE/venv/bin/python"

for p in 1 2 3 4 5 6; do
    script=$(ls "$HERE"/scripts/phase${p}_*.py)
    echo ""
    echo "##################  PHASE $p  ##################"
    "$PY" "$script"
done

echo ""
echo "All phases complete. See results/ and plots/."
