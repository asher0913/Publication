#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_frozen_matrix(selection: dict) -> dict:
    selected = selection["selected"]
    experiments = []
    for seed in range(125, 130):
        experiments.extend(
            [
                {
                    "name": f"seed{seed}_no_defense",
                    "overrides": {"seed": seed, "defense.name": "none"},
                },
                {
                    "name": f"seed{seed}_gaussian",
                    "overrides": {
                        "seed": seed,
                        "defense.name": "gaussian",
                        "defense.noise_std": selected["gaussian"]["noise_std"],
                    },
                },
                {
                    "name": f"seed{seed}_official_cem",
                    "overrides": {
                        "seed": seed,
                        "defense.name": "official_cem",
                        "defense.noise_std": selected["official_cem"]["noise_std"],
                        "defense.privacy_weight": selected["official_cem"][
                            "privacy_weight"
                        ],
                    },
                },
                {
                    "name": f"seed{seed}_single_prototype",
                    "overrides": {
                        "seed": seed,
                        "defense.noise_std": selected["proposed"]["noise_std"],
                        "defense.privacy_weight": selected["proposed"][
                            "privacy_weight"
                        ],
                        "defense.regularizer.num_slots": 1,
                    },
                },
                {
                    "name": f"seed{seed}_slots8",
                    "overrides": {
                        "seed": seed,
                        "defense.noise_std": selected["proposed"]["noise_std"],
                        "defense.privacy_weight": selected["proposed"][
                            "privacy_weight"
                        ],
                    },
                },
            ]
        )
    return {
        "schema_version": 1,
        "frozen_from_test_blind_selection": True,
        "results_root": "results/facescrub/headline_frozen",
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("target_test_accessed") is not False:
        raise ValueError("refusing to freeze a matrix from test-informed selection")
    matrix = build_frozen_matrix(selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"wrote {len(matrix['experiments'])} frozen headline runs to {args.output}")


if __name__ == "__main__":
    main()
