#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_dataset(root: Path) -> None:
    generator = np.random.default_rng(125)
    for split, count in (("train", 6), ("val", 3)):
        for class_index, class_name in enumerate(("class_a", "class_b")):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                base = np.zeros((64, 64, 3), dtype=np.uint8)
                base[..., class_index] = 120 + 20 * image_index
                noise = generator.integers(0, 30, size=base.shape, dtype=np.uint8)
                image = np.clip(base + noise, 0, 255).astype(np.uint8)
                Image.fromarray(image).save(class_dir / f"image_{image_index:02d}.png")


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def last_jsonl(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def finite_metrics(payload: dict, names: tuple[str, ...]) -> bool:
    return all(name in payload and math.isfinite(float(payload[name])) for name in names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "pipeline_smoke.json",
    )
    args = parser.parse_args()

    evidence = {"schema_version": 1, "methods": {}}
    with tempfile.TemporaryDirectory(prefix="publication-cem-smoke-") as temporary:
        temporary_root = Path(temporary)
        data_root = temporary_root / "data"
        results_root = temporary_root / "results"
        write_synthetic_dataset(data_root)

        for defense_name in ("prototype_cem", "epochwise_cem", "official_cem"):
            run_dir = results_root / defense_name
            config = json.loads(
                (ROOT / "configs" / "facescrub_base.json").read_text(encoding="utf-8")
            )
            config.update(seed=125, device="cpu", output_dir=str(run_dir))
            config["data"].update(
                root=str(data_root),
                batch_size=4,
                workers=0,
                image_size=64,
                split_manifest=None,
            )
            config["model"].update(bottleneck_channels=4, pretrained=False)
            config["training"].update(epochs=1, learning_rate=0.001)
            config["attack"].update(
                seed=10125,
                auxiliary_root=str(data_root),
                epochs=1,
                batch_size=4,
                decoder_width=16,
            )
            config["attack"]["evaluation"]["lpips"]["enabled"] = False
            config["attack"]["evaluation"]["face_identity"]["enabled"] = False
            config["defense"].update(name=defense_name, noise_std=0.025)
            config["defense"]["regularizer"].update(
                num_slots=2, projection_dim=16, prototype_momentum=0.9
            )
            config["defense"]["epochwise"].update(
                num_clusters=2, kmeans_iterations=3
            )
            config_path = temporary_root / f"{defense_name}.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            target_stdout = run(
                [sys.executable, "scripts/train_target.py", "--config", str(config_path)]
            )
            attack_dir = run_dir / "attacks" / "seed10125"
            attack_stdout = run(
                [
                    sys.executable,
                    "scripts/train_attack.py",
                    "--checkpoint",
                    str(run_dir / "checkpoint_best.pt"),
                    "--output-dir",
                    str(attack_dir),
                    "--seed",
                    "10125",
                ]
            )
            profile_stdout = run(
                [
                    sys.executable,
                    "scripts/profile_checkpoint.py",
                    "--checkpoint",
                    str(run_dir / "checkpoint_best.pt"),
                    "--output",
                    str(run_dir / "efficiency_profile.json"),
                    "--device",
                    "cpu",
                    "--warmup-batches",
                    "0",
                    "--profile-batches",
                    "1",
                ]
            )

            target = last_jsonl(run_dir / "training.jsonl")
            attack = json.loads(
                (attack_dir / "attack_metrics.json").read_text(encoding="utf-8")
            )
            target_files = sorted(path.name for path in run_dir.iterdir() if path.is_file())
            attack_files = sorted(
                path.name for path in attack_dir.iterdir() if path.is_file()
            )
            evidence["methods"][defense_name] = {
                "target_files": target_files,
                "attack_files": attack_files,
                "target_metrics": target,
                "attack_metrics": attack,
                "target_finite": finite_metrics(
                    target,
                    ("train_task_loss", "train_total_loss", "validation_accuracy"),
                ),
                "attack_finite": finite_metrics(attack, ("mse", "ssim", "psnr")),
                "target_stdout_last": target_stdout.strip().splitlines()[-1],
                "attack_stdout_last": attack_stdout.strip().splitlines()[-1],
                "profile_stdout_last": profile_stdout.strip().splitlines()[-1],
                "profile": json.loads(
                    (run_dir / "efficiency_profile.json").read_text(encoding="utf-8")
                ),
            }

        aggregate_prefix = results_root / "summary"
        aggregate_stdout = run(
            [
                sys.executable,
                "scripts/aggregate_results.py",
                "--results-root",
                str(results_root),
                "--output-prefix",
                str(aggregate_prefix),
            ]
        )
        evidence["aggregation"] = {
            "stdout": aggregate_stdout.strip(),
            "summary": json.loads(
                aggregate_prefix.with_suffix(".json").read_text(encoding="utf-8")
            ),
            "attack_level_csv": aggregate_prefix.with_suffix(".csv").is_file(),
            "target_level_csv": aggregate_prefix.with_name(
                aggregate_prefix.name + "_target_level"
            ).with_suffix(".csv").is_file(),
        }

    evidence["status"] = (
        "PASS"
        if all(
            method["target_finite"] and method["attack_finite"]
            for method in evidence["methods"].values()
        )
        and evidence["aggregation"]["attack_level_csv"]
        and evidence["aggregation"]["target_level_csv"]
        else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
