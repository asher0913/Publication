#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from publication_cem.data import build_protocol_loaders  # noqa: E402
from publication_cem.models import build_split_model  # noqa: E402
from publication_cem.reproducibility import save_json, seed_everything  # noqa: E402
from train_target import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    loaders = build_protocol_loaders(config["data"], seed)
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model = build_split_model(
        model_config,
        num_classes=loaders.num_classes,
        image_size=int(config["data"]["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    defense = config["defense"]
    noise_std = 0.0 if defense["name"] == "none" else float(defense["noise_std"])
    evaluation_seed = seed + 200_000
    metrics = evaluate(model, loaders.test, device, noise_std, evaluation_seed)
    payload = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_split": "target_validation",
        "evaluation_split": "target_test",
        "evaluation_noise_seed": evaluation_seed,
        "target_test_accessed": True,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
    }
    output = args.output or args.checkpoint.parent / "target_test_metrics.json"
    save_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
