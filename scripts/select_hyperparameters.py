#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def family(config: dict) -> str:
    name = config["defense"]["name"]
    return "proposed" if name == "prototype_cem" else name


def collect_candidates(results_root: Path) -> list[dict]:
    candidates = []
    for config_path in sorted(results_root.rglob("resolved_config.json")):
        run_dir = config_path.parent
        selection_path = run_dir / "selection_attack" / "attack_selection_metrics.json"
        target_selection_path = run_dir / "target_selection_metrics.json"
        if not all(
            path.is_file()
            for path in (selection_path, target_selection_path)
        ):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        attack = json.loads(selection_path.read_text(encoding="utf-8"))
        target_selection = json.loads(
            target_selection_path.read_text(encoding="utf-8")
        )
        if attack.get("target_test_accessed") is not False:
            raise ValueError(f"selection result is not test-blind: {selection_path}")
        if target_selection.get("target_test_accessed") is not False:
            raise ValueError(
                f"target selection result is not test-blind: {target_selection_path}"
            )
        candidates.append(
            {
                "run_dir": str(run_dir),
                "family": family(config),
                "validation_accuracy": float(
                    target_selection["best_validation_accuracy"]
                ),
                "attacker_validation_mse": float(
                    attack["best_auxiliary_validation_mse"]
                ),
                "noise_std": float(config["defense"].get("noise_std", 0.0)),
                "privacy_weight": float(
                    config["defense"].get("privacy_weight", 0.0)
                ),
            }
        )
    return candidates


def choose(candidates: list[dict], utility_drop_limit: float) -> dict:
    no_defense = [row for row in candidates if row["family"] == "none"]
    if len(no_defense) != 1:
        raise ValueError("selection requires exactly one no-defence reference")
    reference_accuracy = no_defense[0]["validation_accuracy"]
    threshold = reference_accuracy - utility_drop_limit
    grouped = defaultdict(list)
    for row in candidates:
        if row["family"] != "none" and row["validation_accuracy"] >= threshold:
            grouped[row["family"]].append(row)
    selected = {}
    for name in ("gaussian", "official_cem", "proposed"):
        if not grouped[name]:
            raise ValueError(f"no eligible {name} candidate meets the utility constraint")
        selected[name] = max(
            grouped[name],
            key=lambda row: (
                row["attacker_validation_mse"],
                row["validation_accuracy"],
            ),
        )
    return {
        "reference_validation_accuracy": reference_accuracy,
        "minimum_eligible_accuracy": threshold,
        "utility_drop_limit": utility_drop_limit,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--utility-drop-limit", type=float, default=0.01)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = collect_candidates(args.results_root)
    selection = choose(candidates, args.utility_drop_limit)
    payload = {
        "schema_version": 1,
        "selection_split": "target_validation + attacker_auxiliary_validation",
        "target_test_accessed": False,
        "candidate_count": len(candidates),
        **selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
