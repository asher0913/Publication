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
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-path-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--semantic-noise", nargs="+", type=float, required=True)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--utility-threshold", type=float, default=0.8033)
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_strongest_passing_noise(
    candidates: list[dict], threshold: float, minimum_margin: float
) -> dict | None:
    required_accuracy = threshold + minimum_margin
    passing = [
        candidate
        for candidate in candidates
        if float(candidate["minimum_fused_accuracy"]) > required_accuracy
    ]
    if not passing:
        return None
    return max(passing, key=lambda candidate: float(candidate["semantic_noise_std"]))


def main() -> None:
    args = parse_args()
    if not args.semantic_noise or min(args.semantic_noise) < 0:
        raise ValueError("semantic noise candidates must be non-negative")
    if not args.evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")
    if args.minimum_margin < 0:
        raise ValueError("minimum margin must be non-negative")

    seed_everything(args.evaluation_seeds[0])
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.dual_path_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_args = checkpoint["args"]
    _, validation_loader, class_to_index = build_loaders(
        args.data_root,
        args.batch_size,
        args.workers,
        args.evaluation_seeds[0],
    )
    if class_to_index != checkpoint["class_to_index"]:
        raise ValueError("checkpoint and validation class mappings differ")

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

    candidates = []
    for noise_std in sorted(set(args.semantic_noise)):
        runs = []
        for evaluation_seed in args.evaluation_seeds:
            metrics = evaluate(
                legacy,
                semantic,
                fusion,
                validation_loader,
                device,
                float(checkpoint_args["legacy_noise_std"]),
                noise_std,
                evaluation_seed,
            )
            runs.append(
                {
                    "seed": evaluation_seed,
                    "fused_accuracy": metrics["fused_accuracy"],
                    "semantic_accuracy": metrics["semantic_accuracy"],
                    "legacy_accuracy": metrics["legacy_accuracy"],
                }
            )
        fused_accuracies = [run["fused_accuracy"] for run in runs]
        candidate = {
            "semantic_noise_std": noise_std,
            "mean_fused_accuracy": float(np.mean(fused_accuracies)),
            "standard_deviation_fused_accuracy": float(
                np.std(fused_accuracies, ddof=1) if len(fused_accuracies) > 1 else 0.0
            ),
            "minimum_fused_accuracy": min(fused_accuracies),
            "maximum_fused_accuracy": max(fused_accuracies),
            "runs": runs,
        }
        candidates.append(candidate)
        print(json.dumps(candidate, sort_keys=True), flush=True)

    selected = select_strongest_passing_noise(
        candidates, args.utility_threshold, args.minimum_margin
    )
    result = {
        "status": "PASS" if selected is not None else "FAIL",
        "selection_rule": "largest semantic noise whose minimum repeated accuracy strictly exceeds the required accuracy",
        "published_accuracy_threshold": args.utility_threshold,
        "minimum_accuracy_margin": args.minimum_margin,
        "required_accuracy": args.utility_threshold + args.minimum_margin,
        "legacy_noise_std": float(checkpoint_args["legacy_noise_std"]),
        "checkpoint_training_semantic_noise_std": float(
            checkpoint_args["semantic_noise_std"]
        ),
        "dual_path_checkpoint": str(args.dual_path_checkpoint),
        "evaluation_seeds": args.evaluation_seeds,
        "candidates": candidates,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
