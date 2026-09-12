#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scripts.run_recoverable_jobs import completed
except ModuleNotFoundError:
    from run_recoverable_jobs import completed


def finite_json(path: Path, keys: tuple[str, ...]) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return all(key in payload and math.isfinite(float(payload[key])) for key in keys)


def count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def finite_attack(path: Path, *, require_identity: bool) -> bool:
    keys = ["mse", "ssim", "psnr", "lpips"]
    if require_identity:
        keys.extend(
            (
                "identity_top1_success",
                "face_cosine_similarity",
                "true_identity_prototype_similarity",
                "verification_tar_at_far",
            )
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            all(key in payload and math.isfinite(float(payload[key])) for key in keys)
            and payload.get("evaluation_split") == "target_test"
            and (path.parent / "per_image_metrics.jsonl").stat().st_size > 0
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/uacem_formal")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/uacem_formal/audit.json")
    )
    args = parser.parse_args()
    main_root = args.root / "main"
    general_root = args.root / "generalisation"
    checks = {
        "main_target_tests": count(main_root, "seed12[5-9]_*/target_test_metrics.json") == 20,
        "main_profiles": count(main_root, "seed12[5-9]_*/efficiency_profile.json") == 20,
        "main_attacks": count(main_root, "seed12[5-9]_*/attacks/*/seed*/attack_metrics.json") == 150,
        "main_calibrations": count(main_root, "seed12[5-9]_uacem/channel_calibration.json") == 5,
        "generalisation_target_tests": count(general_root, "*/seed12[5-7]_*/target_test_metrics.json") == 24,
        "generalisation_profiles": count(general_root, "*/seed12[5-7]_*/efficiency_profile.json") == 24,
        "generalisation_attacks": count(general_root, "*/seed12[5-7]_*/attacks/*/seed*/attack_metrics.json") == 72,
        "generalisation_calibrations": count(general_root, "*/seed12[5-7]_uacem/channel_calibration.json") == 12,
        "ablation_variants": count(args.root / "ablation", "*/channel_calibration.json") == 5,
        "ablation_attacks": count(args.root / "ablation", "*/attacks/conv_decoder/seed*/attack_selection_metrics.json") == 15,
    }
    test_paths = list(main_root.glob("seed12[5-9]_*/target_test_metrics.json"))
    attack_paths = list(main_root.glob("seed12[5-9]_*/attacks/*/seed*/attack_metrics.json"))
    checks["finite_main_target_metrics"] = all(
        finite_json(path, ("test_accuracy", "test_loss")) for path in test_paths
    )
    checks["finite_main_attack_metrics"] = all(
        finite_attack(path, require_identity=True) for path in attack_paths
    )
    general_attack_paths = list(
        general_root.glob("*/seed12[5-7]_*/attacks/*/seed*/attack_metrics.json")
    )
    checks["finite_generalisation_attack_metrics"] = all(
        finite_attack(
            path,
            require_identity="facescrub" in path.parts[-6],
        )
        for path in general_attack_paths
    )
    runbook_dir = Path("generated_runbooks/uacem_formal")
    summary = json.loads((runbook_dir / "summary.json").read_text(encoding="utf-8"))
    non_output_jobs = []
    for path in sorted((runbook_dir / "jobs").glob("*.json")):
        if path.stem == "paper_outputs":
            continue
        non_output_jobs.extend(json.loads(path.read_text(encoding="utf-8"))["jobs"])
    checks["frozen_runbook_job_count"] = (
        summary.get("total_jobs") == 419 and len(non_output_jobs) == 410
    )
    checks["all_experiment_jobs_complete"] = all(completed(job) for job in non_output_jobs)
    required_outputs = (
        args.root / "main_summary_target_level.csv",
        args.root / "generalisation_summary_target_level.csv",
        args.root / "statistical_analysis.json",
        args.root / "ablation_summary.json",
        Path("paper/generated/uacem/results_table.tex"),
        Path("paper/generated/uacem/privacy_utility.pdf"),
        Path("paper/generated/uacem/reconstruction_grid.png"),
        Path("paper/generated/uacem/generalisation_table.tex"),
        Path("paper/generated/uacem/ablation_table.tex"),
        Path("paper/generated/uacem/channel_allocation.pdf"),
        Path("paper/generated/uacem/efficiency_table.tex"),
        Path("paper/generated/uacem/provenance.json"),
    )
    checks["paper_outputs"] = all(path.is_file() for path in required_outputs)
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
