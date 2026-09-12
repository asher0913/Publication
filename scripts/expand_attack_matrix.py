#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


def build_commands(
    target_matrix: dict,
    attacker_seeds: list[int],
    name_pattern: str,
    attack_types: list[str] | None = None,
) -> list[str]:
    attack_types = attack_types or ["conv_decoder"]
    matcher = re.compile(name_pattern)
    results_root = Path(target_matrix["results_root"])
    commands = []
    for experiment in target_matrix["experiments"]:
        name = experiment["name"]
        if not matcher.fullmatch(name):
            continue
        checkpoint = results_root / name / "checkpoint_best.pt"
        for attack_type in attack_types:
            for seed in attacker_seeds:
                output_dir = (
                    results_root / name / "attacks" / attack_type / f"seed{seed}"
                )
                commands.append(
                    "python scripts/train_attack.py "
                    f"--checkpoint {shlex.quote(str(checkpoint))} "
                    f"--output-dir {shlex.quote(str(output_dir))} "
                    f"--attack-type {shlex.quote(attack_type)} "
                    f"--seed {seed}"
                )
    return commands


def build_protocol_commands(target_matrix: dict, protocol: dict) -> list[str]:
    commands = []
    for tier in protocol["tiers"]:
        commands.extend(
            build_commands(
                target_matrix,
                [int(seed) for seed in protocol["attacker_seeds"]],
                tier["target_name_regex"],
                list(tier["attack_types"]),
            )
        )
    if len(commands) != len(set(commands)):
        raise ValueError("attack protocol produces duplicate commands")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-matrix", required=True, type=Path)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--attack-type",
        action="append",
        choices=("conv_decoder", "residual_decoder", "gan", "adaptive"),
    )
    parser.add_argument("--name-regex", default=".*")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    matrix = json.loads(args.target_matrix.read_text(encoding="utf-8"))
    if args.protocol is not None:
        if args.seed or args.attack_type or args.name_regex != ".*":
            raise ValueError("--protocol cannot be combined with manual attack filters")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        commands = build_protocol_commands(matrix, protocol)
    else:
        if not args.seed:
            raise ValueError("at least one --seed is required without --protocol")
        commands = build_commands(matrix, args.seed, args.name_regex, args.attack_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {len(commands)} attack commands to {args.output}")


if __name__ == "__main__":
    main()
