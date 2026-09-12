#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/expand_matrix.py \
  --base configs/facescrub_base.json \
  --matrix configs/facescrub_core_matrix.json \
  --output-dir /tmp/publication-cem-generated-configs >/dev/null
python scripts/expand_attack_matrix.py \
  --target-matrix configs/facescrub_core_matrix.json \
  --protocol configs/facescrub_attack_protocol.json \
  --output /tmp/publication-cem-attack-commands.txt >/dev/null
test "$(wc -l < /tmp/publication-cem-attack-commands.txt | tr -d ' ')" = "165"

python - <<'PY'
import json
from pathlib import Path

from scripts.generate_runbooks import generalisation_attack_commands

root = Path.cwd()
protocol = json.loads(
    (root / "configs/generalisation_attack_protocol.json").read_text(encoding="utf-8")
)
commands = []
targets = 0
for dataset in ("facescrub", "cifar100", "cifar10"):
    matrix = json.loads(
        (root / f"configs/{dataset}_generalisation_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    targets += len(matrix["experiments"])
    commands.extend(generalisation_attack_commands(matrix, protocol))
assert targets == protocol["expected_target_runs"] == 30
assert len(commands) == protocol["expected_attack_runs"] == 90
assert len(commands) == len(set(commands))
PY

if command -v latexmk >/dev/null 2>&1 && test -f paper/main.tex; then
  PAPER_OUTPUT="$(mktemp -d /tmp/publication-cem-paper.XXXXXX)"
  (
    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir="${PAPER_OUTPUT}" main.tex >/dev/null
  )
fi

python scripts/build_manifest.py
