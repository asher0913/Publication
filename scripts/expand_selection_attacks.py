#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def build_selection_commands(matrix: dict, attack_seed: int) -> list[str]:
    results_root = Path(matrix["results_root"])
    return [
        "python scripts/train_attack.py "
        f"--checkpoint {shlex.quote(str(results_root / experiment['name'] / 'checkpoint_best.pt'))} "
        f"--output-dir {shlex.quote(str(results_root / experiment['name'] / 'selection_attack'))} "
        "--attack-type conv_decoder "
        f"--seed {attack_seed} --selection-only"
        for experiment in matrix["experiments"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--attack-seed", type=int, default=10_120)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    commands = build_selection_commands(matrix, args.attack_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {len(commands)} selection-only attack commands to {args.output}")


if __name__ == "__main__":
    main()
