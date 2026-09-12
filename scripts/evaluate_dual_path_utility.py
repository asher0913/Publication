#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.dual_path import (  # noqa: E402
    CalibratedLogitFusion,
    GlobalSemanticBottleneck,
)
from publication_cem.semantic_backbones import (  # noqa: E402
    build_semantic_backbone,
    checkpoint_backbone_name,
)
from scripts.train_dual_path_facescrub import (  # noqa: E402
    build_loaders,
    evaluate,
    load_legacy,
    restore_legacy_server,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-path-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-noise-std", required=True, type=float)
    parser.add_argument("--semantic-noise-std", required=True, type=float)
    parser.add_argument("--evaluation-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--utility-threshold", type=float, default=0.8033)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def aggregate_runs(runs: list[dict], threshold: float) -> dict:
    if not runs:
        raise ValueError("at least one utility run is required")
    accuracies = [float(run["fused_accuracy"]) for run in runs]
    conservative = min(accuracies)
    return {
        "status": "PASS" if conservative > threshold else "FAIL",
        "published_accuracy_threshold": threshold,
        "strictly_exceeds_published_accuracy": conservative > threshold,
        "conservative_repeated_validation_accuracy": conservative,
        "mean_repeated_validation_accuracy": float(np.mean(accuracies)),
        "standard_deviation_repeated_validation_accuracy": float(
            np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
        ),
        "minimum_repeated_validation_accuracy": conservative,
        "maximum_repeated_validation_accuracy": max(accuracies),
        "utility_runs": runs,
    }


def main() -> None:
    args = parse_args()
    if min(args.legacy_noise_std, args.semantic_noise_std) < 0:
        raise ValueError("noise standard deviations must be non-negative")
    seed_everything(args.evaluation_seeds[0])
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.dual_path_checkpoint, map_location="cpu", weights_only=False
    )
    _, validation_loader, class_to_index = build_loaders(
        args.data_root, args.batch_size, args.workers, args.evaluation_seeds[0]
    )
    if class_to_index != checkpoint["class_to_index"]:
        raise ValueError("checkpoint and validation class mappings differ")

    checkpoint_args = checkpoint["args"]
    legacy = load_legacy(Path(checkpoint["legacy_checkpoint_dir"]), device)
    restore_legacy_server(legacy, checkpoint)
    backbone, backbone_channels = build_semantic_backbone(
        checkpoint_backbone_name(checkpoint_args), pretrained=False
    )
    semantic = GlobalSemanticBottleneck(
        backbone,
        backbone_channels=backbone_channels,
        token_dim=int(checkpoint_args["token_dim"]),
        num_classes=530,
        dropout=0.1,
    ).to(device)
    semantic.load_state_dict(checkpoint["semantic_state"])
    fusion = CalibratedLogitFusion().to(device)
    fusion.load_state_dict(checkpoint["fusion_state"])

    runs = []
    for evaluation_seed in args.evaluation_seeds:
        metrics = evaluate(
            legacy,
            semantic,
            fusion,
            validation_loader,
            device,
            args.legacy_noise_std,
            args.semantic_noise_std,
            evaluation_seed,
        )
        runs.append({"seed": evaluation_seed, **metrics})
    result = {
        **aggregate_runs(runs, args.utility_threshold),
        "dual_path_checkpoint": str(args.dual_path_checkpoint),
        "effective_legacy_noise_std": args.legacy_noise_std,
        "effective_semantic_noise_std": args.semantic_noise_std,
        "evaluation_seeds": args.evaluation_seeds,
        "target_seed": int(checkpoint_args["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
    }
    save_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
