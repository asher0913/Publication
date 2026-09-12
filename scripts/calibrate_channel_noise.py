#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_protocol_loaders
from publication_cem.models import build_split_model
from publication_cem.reproducibility import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument(
        "--score",
        choices=("reconstruction_to_task", "reconstruction_only", "inverse_task"),
        default="reconstruction_to_task",
    )
    parser.add_argument("--minimum-multiplier", type=float, default=0.5)
    parser.add_argument("--maximum-multiplier", type=float, default=2.0)
    return parser.parse_args()


def channel_scores(channels: list[dict], score: str) -> torch.Tensor:
    if score == "reconstruction_to_task":
        values = [record["reconstruction_to_task_ratio"] for record in channels]
    elif score == "reconstruction_only":
        values = [record["reconstruction_gradient_share"] for record in channels]
    elif score == "inverse_task":
        values = [
            1.0 / max(float(record["task_gradient_share"]), 1e-12)
            for record in channels
        ]
    else:
        raise ValueError(f"unsupported calibration score: {score}")
    scores = torch.tensor(values, dtype=torch.float32)
    return scores / scores.median().clamp_min(1e-12)


def calibrated_multipliers(
    ratios: torch.Tensor,
    alpha: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid multiplier bounds")
    if ratios.ndim != 1 or torch.any(ratios <= 0):
        raise ValueError("ratios must be a positive vector")
    multipliers = ratios.pow(alpha).clamp(minimum, maximum)
    return multipliers / multipliers.square().mean().sqrt()


def multipliers_to_logits(
    multipliers: torch.Tensor, max_log_ratio: float
) -> torch.Tensor:
    if max_log_ratio <= 0:
        raise ValueError("max_log_ratio must be positive")
    if multipliers.ndim != 1 or torch.any(multipliers <= 0):
        raise ValueError("multipliers must be a positive vector")
    if not torch.allclose(
        multipliers.square().mean(),
        torch.ones((), dtype=multipliers.dtype, device=multipliers.device),
        atol=1e-5,
    ):
        raise ValueError("multipliers must have unit RMS")
    bounded = torch.log(multipliers) / max_log_ratio
    if torch.any(bounded.abs() >= 1):
        raise ValueError("requested multipliers exceed the configured log-ratio bound")
    return torch.atanh(bounded)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, noise_std: float, seed: int):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean = model.encode(images)
        transmitted = model.transmit(clean, noise_std, generator)
        logits = model.decode(transmitted)
        total_loss += float(criterion(logits, labels)) * images.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += images.shape[0]
    return {"loss": total_loss / count, "accuracy": correct / count}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    config = copy.deepcopy(checkpoint["config"])
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    channels = diagnostic["channels"]
    scores = channel_scores(channels, args.score)
    multipliers = calibrated_multipliers(
        scores,
        args.alpha,
        args.minimum_multiplier,
        args.maximum_multiplier,
    )

    model_config = config["model"]
    adaptive_config = dict(model_config.get("adaptive_channel_noise", {}))
    adaptive_config.update(enabled=True, learnable=False)
    model_config["adaptive_channel_noise"] = adaptive_config
    model_config.setdefault("margin_head", {"enabled": False})
    config["defense"]["variant"] = "utility_aligned_anisotropic_cem"
    config["defense"]["calibration"] = {
        "source_checkpoint": str(args.checkpoint),
        "diagnostic": str(args.diagnostic),
        "score": args.score,
        "alpha": args.alpha,
        "minimum_multiplier": args.minimum_multiplier,
        "maximum_multiplier": args.maximum_multiplier,
        "fixed_total_noise_power": True,
    }
    config["output_dir"] = str(args.output_dir)

    data_cfg = config["data"]
    loaders = build_protocol_loaders(data_cfg, seed)
    model = build_split_model(
        model_config,
        num_classes=loaders.num_classes,
        image_size=int(data_cfg["image_size"]),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if list(incompatible.unexpected_keys) or list(incompatible.missing_keys) != [
        "channel_noise.allocation_logits"
    ]:
        raise ValueError(
            "source checkpoint is incompatible with calibrated model: "
            f"{incompatible}"
        )
    max_log_ratio = float(adaptive_config.get("max_log_ratio", math.log(2.0)))
    logits = multipliers_to_logits(multipliers.to(device), max_log_ratio)
    with torch.no_grad():
        model.channel_noise.allocation_logits.copy_(logits)
    realised = model.channel_noise.channel_multipliers().detach().cpu()
    if not torch.allclose(realised, multipliers, atol=1e-5):
        raise RuntimeError("channel-noise calibration did not reproduce the target allocation")

    noise_std = float(config["defense"]["noise_std"])
    validation_seed = seed + 100_000
    validation = evaluate(model, loaders.validation, device, noise_std, validation_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "resolved_config.json", config)
    calibration_record = {
        **config["defense"]["calibration"],
        "noise_std": noise_std,
        "channel_multipliers": realised.tolist(),
        "multiplier_rms": float(realised.square().mean().sqrt()),
        "minimum_realised_multiplier": float(realised.min()),
        "maximum_realised_multiplier": float(realised.max()),
        "calibration_seconds": time.perf_counter() - started,
        "validation": validation,
        "selection_split": "target_validation",
        "target_test_accessed": False,
    }
    save_json(args.output_dir / "channel_calibration.json", calibration_record)
    selection_metrics = {
        "best_epoch": 0,
        "best_validation_loss": validation["loss"],
        "best_validation_accuracy": validation["accuracy"],
        "selection_split": "target_validation",
        "evaluation_noise_seed": validation_seed,
        "target_test_accessed": False,
    }
    save_json(args.output_dir / "target_selection_metrics.json", selection_metrics)

    calibrated_checkpoint = copy.deepcopy(checkpoint)
    calibrated_checkpoint.update(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "config": config,
            "validation": validation,
            "calibration": calibration_record,
        }
    )
    torch.save(calibrated_checkpoint, args.output_dir / "checkpoint_best.pt")
    print(json.dumps(calibration_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
