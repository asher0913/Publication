#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.expand_matrix import set_nested
except ModuleNotFoundError:
    from expand_matrix import set_nested


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolved_config(
    base_path: Path,
    overrides: dict[str, Any],
    output_dir: Path,
) -> dict:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        set_nested(config, key, value)
    config["output_dir"] = str(output_dir)
    return config


def job(
    name: str,
    command: list[str],
    completion: Path,
    reset_paths: list[Path],
    completion_json: dict[str, Any] | None = None,
) -> dict:
    payload = {
        "name": name,
        "command": command,
        "completion": str(completion),
        "reset_paths": [str(path) for path in reset_paths],
    }
    if completion_json is not None:
        payload["completion_json"] = completion_json
    return payload


def target_job(name: str, config_path: Path, run_dir: Path) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/train_target.py",
            "--config",
            str(config_path),
            "--selection-only",
        ],
        run_dir / "target_selection_metrics.json",
        [run_dir],
    )


def shadow_job(name: str, checkpoint: Path, run_dir: Path, seed: int) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/train_attack.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run_dir),
            "--attack-type",
            "conv_decoder",
            "--seed",
            str(seed),
            "--selection-only",
        ],
        run_dir / "attack_selection_metrics.json",
        [run_dir],
    )


def diagnostic_job(
    name: str, checkpoint: Path, decoder: Path, output: Path
) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/diagnose_channel_tradeoff.py",
            "--checkpoint",
            str(checkpoint),
            "--decoder",
            str(decoder),
            "--output",
            str(output),
            "--batches",
            "32",
        ],
        output,
        [output],
    )


def calibration_job(
    name: str,
    checkpoint: Path,
    diagnostic: Path,
    run_dir: Path,
    selection: dict,
    *,
    score: str | None = None,
    alpha: float | None = None,
) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/calibrate_channel_noise.py",
            "--checkpoint",
            str(checkpoint),
            "--diagnostic",
            str(diagnostic),
            "--output-dir",
            str(run_dir),
            "--score",
            str(score or selection["score"]),
            "--alpha",
            str(selection["alpha"] if alpha is None else alpha),
            "--minimum-multiplier",
            str(selection["minimum_multiplier"]),
            "--maximum-multiplier",
            str(selection["maximum_multiplier"]),
        ],
        run_dir / "target_selection_metrics.json",
        [run_dir],
    )


def evaluation_job(name: str, run_dir: Path) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/evaluate_target_checkpoint.py",
            "--checkpoint",
            str(run_dir / "checkpoint_best.pt"),
        ],
        run_dir / "target_test_metrics.json",
        [run_dir / "target_test_metrics.json"],
    )


def attack_job(
    name: str,
    run_dir: Path,
    attack_type: str,
    attacker_seed: int,
    *,
    selection_only: bool = False,
) -> dict:
    attack_dir = run_dir / "attacks" / attack_type / f"seed{attacker_seed}"
    command = [
        "python",
        "scripts/train_attack.py",
        "--checkpoint",
        str(run_dir / "checkpoint_best.pt"),
        "--output-dir",
        str(attack_dir),
        "--attack-type",
        attack_type,
        "--seed",
        str(attacker_seed),
    ]
    completion_name = "attack_metrics.json"
    if selection_only:
        command.append("--selection-only")
        completion_name = "attack_selection_metrics.json"
    return job(
        name,
        command,
        attack_dir / completion_name,
        [attack_dir],
    )


def profile_job(name: str, run_dir: Path) -> dict:
    return job(
        name,
        [
            "python",
            "scripts/profile_checkpoint.py",
            "--checkpoint",
            str(run_dir / "checkpoint_best.pt"),
        ],
        run_dir / "efficiency_profile.json",
        [run_dir / "efficiency_profile.json"],
    )


def main_jobs(protocol: dict, output: Path) -> dict[str, list[dict]]:
    selection = protocol["selection"]
    main = protocol["main"]
    base_path = ROOT / "configs/facescrub_base.json"
    results_root = Path("results/uacem_formal/main")
    configs_dir = output / "configs/main"
    jobs: dict[str, list[dict]] = {
        "main_targets": [],
        "main_shadow_attacks": [],
        "main_calibrations": [],
        "main_evaluations": [],
        "main_attacks": [],
        "main_profiles": [],
    }
    method_overrides = {
        "no_defense": {"defense.name": "none"},
        "gaussian": {
            "defense.name": "gaussian",
            "defense.noise_std": selection["noise_std"],
        },
        "official_cem": {
            "defense.name": "official_cem",
            "defense.noise_std": selection["noise_std"],
            "defense.privacy_weight": selection["official_privacy_weight"],
        },
    }
    run_dirs: dict[tuple[int, str], Path] = {}
    for seed in main["target_seeds"]:
        for method in main["control_methods"]:
            run_dir = results_root / f"seed{seed}_{method}"
            overrides = {"seed": seed, **method_overrides[method]}
            config_path = configs_dir / f"seed{seed}_{method}.json"
            write_json(config_path, resolved_config(base_path, overrides, run_dir))
            jobs["main_targets"].append(
                target_job(f"main_seed{seed}_{method}", config_path, run_dir)
            )
            run_dirs[(seed, method)] = run_dir

        official_dir = run_dirs[(seed, "official_cem")]
        shadow_dir = official_dir / "diagnostic_shadow"
        diagnostic = results_root / "diagnostics" / f"seed{seed}.json"
        uacem_dir = results_root / f"seed{seed}_uacem"
        jobs["main_shadow_attacks"].append(
            shadow_job(
                f"main_seed{seed}_shadow",
                official_dir / "checkpoint_best.pt",
                shadow_dir,
                90_000 + seed,
            )
        )
        jobs["main_calibrations"].extend(
            [
                diagnostic_job(
                    f"main_seed{seed}_diagnostic",
                    official_dir / "checkpoint_best.pt",
                    shadow_dir / "decoder.pt",
                    diagnostic,
                ),
                calibration_job(
                    f"main_seed{seed}_uacem",
                    official_dir / "checkpoint_best.pt",
                    diagnostic,
                    uacem_dir,
                    selection,
                ),
            ]
        )
        run_dirs[(seed, "uacem")] = uacem_dir

    all_methods = [*main["control_methods"], "uacem"]
    for seed in main["target_seeds"]:
        for method in all_methods:
            run_dir = run_dirs[(seed, method)]
            jobs["main_evaluations"].append(
                evaluation_job(f"main_seed{seed}_{method}_test", run_dir)
            )
            jobs["main_profiles"].append(
                profile_job(f"main_seed{seed}_{method}_profile", run_dir)
            )
            attack_types = list(main["primary_attack_types"])
            if method in main["strong_attack_methods"]:
                attack_types.extend(main["strong_attack_types"])
            for attack_type in attack_types:
                for attacker_seed in main["attacker_seeds"]:
                    jobs["main_attacks"].append(
                        attack_job(
                            f"main_seed{seed}_{method}_{attack_type}_{attacker_seed}",
                            run_dir,
                            attack_type,
                            attacker_seed,
                        )
                    )
    return jobs


def generalisation_jobs(protocol: dict, output: Path) -> dict[str, list[dict]]:
    selection = protocol["selection"]
    general = protocol["generalisation"]
    root = Path("results/uacem_formal/generalisation")
    configs_dir = output / "configs/generalisation"
    jobs: dict[str, list[dict]] = {
        "generalisation_targets": [],
        "generalisation_shadow_attacks": [],
        "generalisation_calibrations": [],
        "generalisation_evaluations": [],
        "generalisation_attacks": [],
        "generalisation_profiles": [],
    }
    for variant in general["variants"]:
        base_path = ROOT / variant["base"]
        variant_root = root / variant["name"]
        for seed in general["target_seeds"]:
            official_dir = variant_root / f"seed{seed}_official_cem"
            overrides = {
                "seed": seed,
                "defense.name": "official_cem",
                "defense.noise_std": selection["noise_std"],
                "defense.privacy_weight": selection["official_privacy_weight"],
                **variant["overrides"],
            }
            config_path = configs_dir / variant["name"] / f"seed{seed}_official.json"
            write_json(config_path, resolved_config(base_path, overrides, official_dir))
            jobs["generalisation_targets"].append(
                target_job(
                    f"general_{variant['name']}_seed{seed}_official",
                    config_path,
                    official_dir,
                )
            )
            shadow_dir = official_dir / "diagnostic_shadow"
            diagnostic = variant_root / "diagnostics" / f"seed{seed}.json"
            uacem_dir = variant_root / f"seed{seed}_uacem"
            jobs["generalisation_shadow_attacks"].append(
                shadow_job(
                    f"general_{variant['name']}_seed{seed}_shadow",
                    official_dir / "checkpoint_best.pt",
                    shadow_dir,
                    90_000 + seed,
                )
            )
            jobs["generalisation_calibrations"].extend(
                [
                    diagnostic_job(
                        f"general_{variant['name']}_seed{seed}_diagnostic",
                        official_dir / "checkpoint_best.pt",
                        shadow_dir / "decoder.pt",
                        diagnostic,
                    ),
                    calibration_job(
                        f"general_{variant['name']}_seed{seed}_uacem",
                        official_dir / "checkpoint_best.pt",
                        diagnostic,
                        uacem_dir,
                        selection,
                    ),
                ]
            )
            for method, run_dir in (("official_cem", official_dir), ("uacem", uacem_dir)):
                jobs["generalisation_evaluations"].append(
                    evaluation_job(
                        f"general_{variant['name']}_seed{seed}_{method}_test",
                        run_dir,
                    )
                )
                jobs["generalisation_profiles"].append(
                    profile_job(
                        f"general_{variant['name']}_seed{seed}_{method}_profile",
                        run_dir,
                    )
                )
                for attack_type in general["attack_types"]:
                    for attacker_seed in general["attacker_seeds"]:
                        jobs["generalisation_attacks"].append(
                            attack_job(
                                f"general_{variant['name']}_seed{seed}_{method}_{attack_type}_{attacker_seed}",
                                run_dir,
                                attack_type,
                                attacker_seed,
                            )
                        )
    return jobs


def ablation_jobs(protocol: dict) -> dict[str, list[dict]]:
    selection = protocol["selection"]
    ablation = protocol["ablation"]
    source = Path(ablation["source_checkpoint"])
    root = Path("results/uacem_formal/ablation")
    shadow_dir = root / "diagnostic_shadow"
    diagnostic = root / "channel_diagnostic.json"
    jobs = {
        "ablation_setup": [
            shadow_job(
                "ablation_shadow",
                source,
                shadow_dir,
                90_120,
            ),
            diagnostic_job(
                "ablation_diagnostic",
                source,
                shadow_dir / "decoder.pt",
                diagnostic,
            ),
        ],
        "ablation_calibrations": [],
        "ablation_attacks": [],
    }
    for variant in ablation["variants"]:
        run_dir = root / variant["name"]
        jobs["ablation_calibrations"].append(
            calibration_job(
                f"ablation_{variant['name']}",
                source,
                diagnostic,
                run_dir,
                selection,
                score=variant["score"],
                alpha=float(variant["alpha"]),
            )
        )
        for attacker_seed in ablation["attacker_seeds"]:
            jobs["ablation_attacks"].append(
                attack_job(
                    f"ablation_{variant['name']}_{attacker_seed}",
                    run_dir,
                    "conv_decoder",
                    attacker_seed,
                    selection_only=True,
                )
            )
    return jobs


def paper_output_jobs() -> dict[str, list[dict]]:
    root = Path("results/uacem_formal")
    paper = Path("paper/generated/uacem")
    jobs = [
        job(
            "aggregate_main",
            [
                "python",
                "scripts/aggregate_results.py",
                "--results-root",
                str(root / "main"),
                "--output-prefix",
                str(root / "main_summary"),
            ],
            root / "main_summary_target_level.csv",
            [
                root / "main_summary.csv",
                root / "main_summary.json",
                root / "main_summary_target_level.csv",
            ],
        ),
        job(
            "aggregate_generalisation",
            [
                "python",
                "scripts/aggregate_results.py",
                "--results-root",
                str(root / "generalisation"),
                "--output-prefix",
                str(root / "generalisation_summary"),
            ],
            root / "generalisation_summary_target_level.csv",
            [
                root / "generalisation_summary.csv",
                root / "generalisation_summary.json",
                root / "generalisation_summary_target_level.csv",
            ],
        ),
        job(
            "paired_statistics",
            [
                "python",
                "scripts/statistical_analysis.py",
                "--target-level-csv",
                str(root / "main_summary_target_level.csv"),
                "--proposed-pattern",
                "uacem_.*",
                "--baseline-pattern",
                "official_cem_.*",
                "--output",
                str(root / "statistical_analysis.json"),
            ],
            root / "statistical_analysis.json",
            [root / "statistical_analysis.json"],
        ),
        job(
            "summarise_ablation",
            [
                "python",
                "scripts/summarise_uacem_ablation.py",
                "--output",
                str(root / "ablation_summary.json"),
            ],
            root / "ablation_summary.json",
            [root / "ablation_summary.json", root / "ablation_summary.csv"],
        ),
        job(
            "generate_main_assets",
            [
                "python",
                "scripts/generate_paper_assets.py",
                "--target-level-csv",
                str(root / "main_summary_target_level.csv"),
                "--output-dir",
                str(paper),
            ],
            paper / "privacy_utility.pdf",
            [
                paper / "results_table.tex",
                paper / "efficiency_table.tex",
                paper / "privacy_utility.pdf",
                paper / "provenance.json",
            ],
        ),
        job(
            "generate_supplementary_assets",
            [
                "python",
                "scripts/generate_uacem_supplementary_assets.py",
                "--generalisation-csv",
                str(root / "generalisation_summary_target_level.csv"),
                "--ablation-json",
                str(root / "ablation_summary.json"),
                "--main-results-root",
                str(root / "main"),
                "--output-dir",
                str(paper),
            ],
            paper / "channel_allocation.pdf",
            [
                paper / "generalisation_table.tex",
                paper / "ablation_table.tex",
                paper / "channel_allocation.pdf",
            ],
        ),
        job(
            "generate_qualitative_grid",
            [
                "python",
                "scripts/export_qualitative_grid.py",
                "--results-root",
                str(root / "main"),
                "--output",
                str(paper / "reconstruction_grid.png"),
                "--method-set",
                "uacem",
            ],
            paper / "reconstruction_grid.png",
            [
                paper / "reconstruction_grid.png",
                paper / "reconstruction_grid.provenance.json",
            ],
        ),
        job(
            "audit_evidence",
            ["python", "scripts/audit_uacem_results.py"],
            root / "audit.json",
            [root / "audit.json"],
            completion_json={"status": "PASS"},
        ),
        job(
            "build_handoff",
            ["python", "scripts/build_uacem_handoff.py"],
            Path("handoff/uacem_paper_writing_bundle.tar.gz"),
            [
                Path("handoff/uacem_paper_writing_bundle.tar.gz"),
                Path("handoff/uacem_paper_writing_bundle.json"),
            ],
        ),
    ]
    return {"paper_outputs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/uacem_formal_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "generated_runbooks/uacem_formal",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen_protocol = copy.deepcopy(protocol)
    frozen_protocol["source_protocol"] = str(args.protocol.relative_to(ROOT))
    frozen_protocol["source_protocol_sha256"] = file_sha256(args.protocol)
    write_json(args.output_dir / "frozen_protocol.json", frozen_protocol)

    stages = {}
    stages.update(main_jobs(protocol, args.output_dir))
    stages.update(generalisation_jobs(protocol, args.output_dir))
    stages.update(ablation_jobs(protocol))
    stages.update(paper_output_jobs())
    counts = {}
    for stage, jobs in stages.items():
        write_json(args.output_dir / "jobs" / f"{stage}.json", {"jobs": jobs})
        counts[stage] = len(jobs)
    summary = {
        "schema_version": 1,
        "protocol_sha256": file_sha256(args.protocol),
        "stages": counts,
        "total_jobs": sum(counts.values()),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
