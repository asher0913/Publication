#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
from pathlib import Path

try:
    from scripts.expand_attack_matrix import build_protocol_commands
    from scripts.expand_matrix import expand_matrix
    from scripts.expand_selection_attacks import build_selection_commands
except ModuleNotFoundError:
    from expand_attack_matrix import build_protocol_commands
    from expand_matrix import expand_matrix
    from expand_selection_attacks import build_selection_commands


ROOT = Path(__file__).resolve().parents[1]


def write_commands(path: Path, commands: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_generalisation_matrix(matrix: dict, selection: dict) -> dict:
    """Apply primary-dataset validation choices without secondary retuning."""
    frozen = copy.deepcopy(matrix)
    selected = selection["selected"]
    for experiment in frozen["experiments"]:
        overrides = experiment.setdefault("overrides", {})
        defense_name = overrides.get("defense.name", "prototype_cem")
        family = "official_cem" if defense_name == "official_cem" else "proposed"
        overrides["defense.noise_std"] = selected[family]["noise_std"]
        overrides["defense.privacy_weight"] = selected[family]["privacy_weight"]
    frozen["frozen_from_test_blind_selection"] = True
    frozen["selection_policy"] = (
        "FaceScrub validation-selected hyperparameters transferred without "
        "secondary-dataset retuning"
    )
    return frozen


def generalisation_attack_commands(matrix: dict, protocol: dict) -> list[str]:
    return build_protocol_commands(
        matrix,
        {
            "attacker_seeds": protocol["attacker_seeds"],
            "tiers": [
                {
                    "target_name_regex": ".*",
                    "attack_types": protocol["attack_types"],
                }
            ],
        },
    )


def load_audit() -> dict:
    audit_path = ROOT / "results/readiness_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("run the publication readiness audit first")
    return json.loads(audit_path.read_text(encoding="utf-8"))


def pilot_runbook(output_dir: Path) -> dict:
    audit = load_audit()
    if audit.get("code_status") != "CODE_READY":
        raise RuntimeError("formal pilot runbook requires CODE_READY")
    base = json.loads((ROOT / "configs/facescrub_base.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        (ROOT / "configs/facescrub_pilot_matrix.json").read_text(encoding="utf-8")
    )
    targets = expand_matrix(base, matrix, output_dir / "configs", selection_only=True)
    results_root = Path(matrix["results_root"])
    attacks = []
    for experiment in matrix["experiments"]:
        attack_types = (
            ("conv_decoder", "residual_decoder", "gan", "adaptive")
            if experiment["name"] == "seed119_proposed"
            else ("conv_decoder",)
        )
        checkpoint = results_root / experiment["name"] / "checkpoint_best.pt"
        for attack_type in attack_types:
            attack_dir = (
                results_root / experiment["name"] / "pilot_attacks" / attack_type
            )
            attacks.append(
                "python scripts/train_attack.py "
                f"--checkpoint {shlex.quote(str(checkpoint))} "
                f"--output-dir {shlex.quote(str(attack_dir))} "
                f"--attack-type {attack_type} --seed 10119 --selection-only"
            )
    write_commands(output_dir / "01_pilot_targets.txt", targets)
    write_commands(output_dir / "02_pilot_attacks.txt", attacks)
    write_commands(
        output_dir / "03_pilot_audit.txt",
        [
            "python scripts/summarise_formal_pilot.py",
            "python scripts/audit_publication_readiness.py --run-checks --allow-incomplete",
            "python scripts/preflight_formal_run.py --gate pilot",
        ],
    )
    return {
        "pilot_target_commands": len(targets),
        "pilot_attack_commands": len(attacks),
    }


def selection_runbook(output_dir: Path) -> dict:
    audit = load_audit()
    if audit.get("pilot_status") != "PILOT_READY":
        raise RuntimeError("selection runbook requires PILOT_READY")
    base = json.loads((ROOT / "configs/facescrub_base.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        (ROOT / "configs/facescrub_selection_matrix.json").read_text(encoding="utf-8")
    )
    configs_dir = output_dir / "configs"
    targets = expand_matrix(base, matrix, configs_dir, selection_only=True)
    attacks = build_selection_commands(matrix, 10_120)
    write_commands(output_dir / "01_targets.txt", targets)
    write_commands(output_dir / "02_selection_attacks.txt", attacks)
    post = [
        "python scripts/select_hyperparameters.py "
        "--results-root results/facescrub/selection "
        "--output evidence/frozen_hyperparameters.json",
        "python scripts/freeze_headline_matrix.py "
        "--selection evidence/frozen_hyperparameters.json "
        "--output configs/frozen_facescrub_core_matrix.json",
        "python scripts/audit_publication_readiness.py --run-checks --allow-incomplete",
    ]
    write_commands(output_dir / "03_freeze.txt", post)
    return {"target_commands": len(targets), "attack_commands": len(attacks)}


def headline_runbook(output_dir: Path) -> dict:
    audit = load_audit()
    if audit.get("pre_run_status") != "PRE_RUN_READY":
        raise RuntimeError("headline runbook requires PRE_RUN_READY")
    base_path = ROOT / "configs/facescrub_base.json"
    matrix_path = ROOT / "configs/frozen_facescrub_core_matrix.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    selection_path = ROOT / "evidence/frozen_hyperparameters.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    targets = expand_matrix(base, matrix, output_dir / "configs/headline")
    protocol = json.loads(
        (ROOT / "configs/facescrub_attack_protocol.json").read_text(encoding="utf-8")
    )
    attacks = build_protocol_commands(matrix, protocol)
    profiles = [
        "python scripts/profile_checkpoint.py "
        f"--checkpoint {shlex.quote(str(Path(matrix['results_root']) / experiment['name'] / 'checkpoint_best.pt'))}"
        for experiment in matrix["experiments"]
    ]
    write_commands(output_dir / "01_headline_targets.txt", targets)
    write_commands(output_dir / "02_headline_attacks.txt", attacks)
    write_commands(output_dir / "03_headline_profiles.txt", profiles)

    generalisation_protocol = json.loads(
        (ROOT / "configs/generalisation_attack_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    generalisation_count = 0
    generalisation_attack_count = 0
    for dataset in ("facescrub", "cifar100", "cifar10"):
        general_base = json.loads(
            (ROOT / f"configs/{dataset}_base.json").read_text(encoding="utf-8")
        )
        general_matrix = freeze_generalisation_matrix(json.loads(
            (ROOT / f"configs/{dataset}_generalisation_matrix.json").read_text(
                encoding="utf-8"
            )
        ), selection)
        commands = expand_matrix(
            general_base,
            general_matrix,
            output_dir / f"configs/{dataset}_generalisation",
        )
        write_commands(output_dir / f"04_{dataset}_generalisation_targets.txt", commands)
        attack_commands = generalisation_attack_commands(
            general_matrix, generalisation_protocol
        )
        write_commands(
            output_dir / f"04_{dataset}_generalisation_attacks.txt",
            attack_commands,
        )
        generalisation_count += len(commands)
        generalisation_attack_count += len(attack_commands)
    if generalisation_count != int(generalisation_protocol["expected_target_runs"]):
        raise ValueError("generalisation target count differs from frozen protocol")
    if generalisation_attack_count != int(
        generalisation_protocol["expected_attack_runs"]
    ):
        raise ValueError("generalisation attack count differs from frozen protocol")
    provenance = {
        "schema_version": 1,
        "selection_file": str(selection_path.relative_to(ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "target_runs": generalisation_count,
        "attack_runs": generalisation_attack_count,
        "selection_policy": generalisation_protocol["selection_policy"],
    }
    (output_dir / "generalisation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    post = [
        "python scripts/aggregate_results.py "
        f"--results-root {shlex.quote(str(matrix['results_root']))} "
        "--output-prefix results/headline_summary",
        "python scripts/statistical_analysis.py "
        "--target-level-csv results/headline_summary_target_level.csv "
        "--output results/headline_statistical_analysis.json",
        "python scripts/generate_paper_assets.py "
        "--target-level-csv results/headline_summary_target_level.csv "
        "--output-dir paper/generated",
        "python scripts/export_qualitative_grid.py "
        f"--results-root {shlex.quote(str(matrix['results_root']))} "
        "--output paper/generated/reconstruction_grid.png",
    ]
    write_commands(output_dir / "05_aggregate_analyse.txt", post)
    write_commands(
        output_dir / "06_final_audit.txt",
        [
            "python scripts/audit_publication_readiness.py --run-checks --allow-incomplete",
            "python scripts/build_manifest.py",
        ],
    )
    return {
        "headline_target_commands": len(targets),
        "headline_attack_commands": len(attacks),
        "headline_profile_commands": len(profiles),
        "generalisation_target_commands": generalisation_count,
        "generalisation_attack_commands": generalisation_attack_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("pilot", "selection", "headline"), required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "generated_runbooks"
    )
    args = parser.parse_args()
    stage_dir = args.output_dir / args.stage
    try:
        if args.stage == "pilot":
            summary = pilot_runbook(stage_dir)
        elif args.stage == "selection":
            summary = selection_runbook(stage_dir)
        else:
            summary = headline_runbook(stage_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from None
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
