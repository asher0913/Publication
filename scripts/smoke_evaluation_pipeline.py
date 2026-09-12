#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.evaluators import FaceEmbedder
from publication_cem.reproducibility import module_state_sha256
from scripts.smoke_publication_pipeline import run, write_synthetic_dataset


REQUIRED = (
    "mse",
    "ssim",
    "psnr",
    "lpips",
    "identity_top1_success",
    "face_cosine_similarity",
    "verification_tar_at_far",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/evaluation_pipeline_smoke.json",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="publication-evaluation-smoke-") as temporary:
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
        config["training"].update(epochs=1, learning_rate=0.001)
        config["defense"].update(name="none")
        config["attack"].update(
            auxiliary_root=str(data_root),
            epochs=1,
            batch_size=4,
            decoder_width=16,
        )
        config["attack"]["evaluation"]["assets_manifest"] = str(
            ROOT / "evidence/evaluation_assets.json"
        )
        protocol_path = temporary_root / "face_protocol.pt"
        config["attack"]["evaluation"]["face_identity"]["protocol"] = str(
            protocol_path
        )
        embedder = FaceEmbedder("cpu")
        protocol = {
            "schema_version": 1,
            "embedding_model": "facenet-pytorch/InceptionResnetV1-vggface2",
            "embedding_model_weights_sha256": module_state_sha256(embedder.model),
            "class_to_index": {"class_a": 0, "class_b": 1},
            "prototypes": torch.eye(2, 512),
            "target_far": 0.001,
            "verification_threshold": -1.0,
        }
        torch.save(protocol, protocol_path)
        config_path = temporary_root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        run([sys.executable, "scripts/train_target.py", "--config", str(config_path)])
        attack_dir = run_dir / "evaluation_attack"
        run(
            [
                sys.executable,
                "scripts/train_attack.py",
                "--checkpoint",
                str(run_dir / "checkpoint_best.pt"),
                "--output-dir",
                str(attack_dir),
                "--attack-type",
                "conv_decoder",
                "--seed",
                "10125",
            ]
        )
        metrics = json.loads(
            (attack_dir / "attack_metrics.json").read_text(encoding="utf-8")
        )
        first_row = json.loads(
            (attack_dir / "per_image_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        passed = (
            all(name in metrics and math.isfinite(float(metrics[name])) for name in REQUIRED)
            and all(name in first_row for name in REQUIRED[:6])
            and "verification_accept_at_far" in first_row
            and metrics.get("lpips_enabled") is True
            and metrics.get("face_identity_enabled") is True
        )
        evidence = {
            "schema_version": 1,
            "status": "PASS" if passed else "FAIL",
            "purpose": "Evaluator interface smoke only; not a paper result.",
            "required_metrics": list(REQUIRED),
            "metrics": {name: metrics.get(name) for name in REQUIRED},
            "lpips_enabled": metrics.get("lpips_enabled"),
            "face_identity_enabled": metrics.get("face_identity_enabled"),
            "per_image_keys": sorted(first_row),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "output": str(args.output)}))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
