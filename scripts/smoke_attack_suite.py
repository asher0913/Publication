#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_publication_pipeline import run, write_synthetic_dataset


ATTACK_TYPES = ("conv_decoder", "residual_decoder", "gan", "adaptive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/attack_suite_smoke.json",
    )
    args = parser.parse_args()
    evidence = {"schema_version": 1, "attacks": {}}
    with tempfile.TemporaryDirectory(prefix="publication-attack-suite-") as temporary:
        temporary_root = Path(temporary)
        data_root = temporary_root / "data"
        run_dir = temporary_root / "target"
        write_synthetic_dataset(data_root)
        config = json.loads(
            (ROOT / "configs/facescrub_base.json").read_text(encoding="utf-8")
        )
        config.update(seed=125, device="cpu", output_dir=str(run_dir))
        config["data"].update(
            root=str(data_root),
            batch_size=4,
            workers=0,
            split_manifest=None,
        )
        config["model"].update(bottleneck_channels=4, pretrained=False)
        config["training"].update(
            epochs=1, learning_rate=0.001, gradient_diagnostic_interval=1
        )
        config["defense"].update(name="none")
        config["attack"].update(
            auxiliary_root=str(data_root),
            epochs=1,
            batch_size=4,
            decoder_width=16,
            adaptive_eot_samples=2,
        )
        config["attack"]["evaluation"]["lpips"]["enabled"] = False
        config["attack"]["evaluation"]["face_identity"]["enabled"] = False
        config_path = temporary_root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        run([sys.executable, "scripts/train_target.py", "--config", str(config_path)])
        checkpoint = run_dir / "checkpoint_best.pt"
        selection_dir = run_dir / "selection_attack"
        run(
            [
                sys.executable,
                "scripts/train_attack.py",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(selection_dir),
                "--attack-type",
                "conv_decoder",
                "--seed",
                "10120",
                "--selection-only",
            ]
        )
        selection_metrics = json.loads(
            (selection_dir / "attack_selection_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        evidence["selection_only"] = {
            "target_test_accessed": selection_metrics["target_test_accessed"],
            "evaluation_split": selection_metrics["evaluation_split"],
            "finite": math.isfinite(
                float(selection_metrics["best_auxiliary_validation_mse"])
            ),
        }
        for attack_type in ATTACK_TYPES:
            attack_dir = run_dir / "attacks" / attack_type / "seed10125"
            run(
                [
                    sys.executable,
                    "scripts/train_attack.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(attack_dir),
                    "--attack-type",
                    attack_type,
                    "--seed",
                    "10125",
                ]
            )
            metrics = json.loads(
                (attack_dir / "attack_metrics.json").read_text(encoding="utf-8")
            )
            finite = all(
                math.isfinite(float(metrics[name])) for name in ("mse", "ssim", "psnr")
            )
            evidence["attacks"][attack_type] = {
                "finite": finite,
                "attack_type": metrics["attack_type"],
                "best_epoch": metrics["best_epoch"],
                "mse": metrics["mse"],
                "artifacts": sorted(path.name for path in attack_dir.iterdir()),
            }
    evidence["status"] = (
        "PASS"
        if set(evidence["attacks"]) == set(ATTACK_TYPES)
        and all(value["finite"] for value in evidence["attacks"].values())
        and evidence["selection_only"]["target_test_accessed"] is False
        and evidence["selection_only"]["finite"]
        else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
