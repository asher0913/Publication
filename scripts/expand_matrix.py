#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path


def set_nested(mapping, dotted_key, value) -> None:
    keys = dotted_key.split(".")
    current = mapping
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def expand_matrix(
    base: dict,
    matrix: dict,
    output_dir: Path,
    selection_only: bool = False,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for experiment in matrix["experiments"]:
        resolved = copy.deepcopy(base)
        for dotted_key, value in experiment["overrides"].items():
            set_nested(resolved, dotted_key, value)
        resolved["output_dir"] = str(
            Path(matrix["results_root"]) / experiment["name"]
        )
        config_path = output_dir / f"{experiment['name']}.json"
        config_path.write_text(
            json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
        )
        command = (
            "python scripts/train_target.py --config "
            f"{shlex.quote(str(config_path))}"
        )
        if selection_only:
            command += " --selection-only"
        commands.append(command)
    (output_dir / "commands.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    commands = expand_matrix(base, matrix, args.output_dir)
    print("\n".join(commands))


if __name__ == "__main__":
    main()
