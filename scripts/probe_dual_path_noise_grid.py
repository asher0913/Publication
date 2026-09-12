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
from publication_cem.dual_path_attack import DualPathReconstructor  # noqa: E402
from scripts.attack_dual_path_facescrub import (  # noqa: E402
    build_attack_loaders,
    evaluate as evaluate_attack,
)
from scripts.train_dual_path_facescrub import (  # noqa: E402
    build_loaders,
    evaluate as evaluate_utility,
    load_legacy,
    restore_legacy_server,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-path-checkpoint", required=True, type=Path)
    parser.add_argument("--attacker-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-noise", nargs="+", type=float, required=True)
    parser.add_argument("--semantic-noise", nargs="+", type=float, required=True)
    parser.add_argument("--utility-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--attack-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--utility-threshold", type=float, default=0.8033)
    parser.add_argument("--utility-margin", type=float, default=0.005)
    parser.add_argument("--training-mse-threshold", type=float, default=0.0182)
    parser.add_argument("--inference-mse-threshold", type=float, default=0.0211)
    parser.add_argument(
        "--probe-all",
        action="store_true",
        help="evaluate attacker leakage even when the utility screen fails",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_probe_candidate(candidates: list[dict]) -> dict | None:
    passing = [candidate for candidate in candidates if candidate["probe_pass"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda candidate: (
            float(candidate["minimum_normalised_privacy_margin"]),
            float(candidate["minimum_fused_accuracy"]),
        ),
    )


def load_models(
    target_checkpoint_path: Path,
    attacker_checkpoint_path: Path,
    device: torch.device,
):
    target = torch.load(
        target_checkpoint_path, map_location="cpu", weights_only=False
    )
    target_args = target["args"]
    legacy = load_legacy(Path(target["legacy_checkpoint_dir"]), device)
    restore_legacy_server(legacy, target)
    backbone, backbone_channels = build_semantic_backbone(
        checkpoint_backbone_name(target_args), pretrained=False
    )
    semantic = GlobalSemanticBottleneck(
        backbone,
        backbone_channels=backbone_channels,
        token_dim=int(target_args["token_dim"]),
        num_classes=530,
        dropout=0.1,
    ).to(device)
    semantic.load_state_dict(target["semantic_state"])
    semantic.eval().requires_grad_(False)
    fusion = CalibratedLogitFusion().to(device)
    fusion.load_state_dict(target["fusion_state"])
    fusion.eval().requires_grad_(False)

    attacker = torch.load(
        attacker_checkpoint_path, map_location="cpu", weights_only=False
    )
    attacker_args = attacker["args"]
    reconstructor = DualPathReconstructor(
        semantic_dim=int(target_args["token_dim"]),
        width=int(attacker_args["width"]),
        residual_blocks=int(attacker_args["residual_blocks"]),
    ).to(device)
    reconstructor.load_state_dict(attacker["reconstructor_state"])
    reconstructor.eval().requires_grad_(False)
    return target, legacy, semantic, fusion, reconstructor


def main() -> None:
    args = parse_args()
    if min(args.legacy_noise + args.semantic_noise) < 0:
        raise ValueError("noise candidates must be non-negative")
    if args.utility_margin < 0:
        raise ValueError("utility margin must be non-negative")
    seed_everything(args.utility_seeds[0])
    device = torch.device(args.device)
    target, legacy, semantic, fusion, reconstructor = load_models(
        args.dual_path_checkpoint, args.attacker_checkpoint, device
    )
    _, validation_loader, class_to_index = build_loaders(
        args.data_root, args.batch_size, args.workers, args.utility_seeds[0]
    )
    if class_to_index != target["class_to_index"]:
        raise ValueError("checkpoint and validation class mappings differ")
    attack_training_loader, attack_inference_loader = build_attack_loaders(
        args.data_root, args.batch_size, args.workers, "inference"
    )

    required_accuracy = args.utility_threshold + args.utility_margin
    candidates = []
    for legacy_noise in sorted(set(args.legacy_noise)):
        for semantic_noise in sorted(set(args.semantic_noise)):
            utility_runs = [
                evaluate_utility(
                    legacy,
                    semantic,
                    fusion,
                    validation_loader,
                    device,
                    legacy_noise,
                    semantic_noise,
                    seed,
                )["fused_accuracy"]
                for seed in args.utility_seeds
            ]
            candidate = {
                "legacy_noise_std": legacy_noise,
                "semantic_noise_std": semantic_noise,
                "mean_fused_accuracy": float(np.mean(utility_runs)),
                "minimum_fused_accuracy": min(utility_runs),
                "maximum_fused_accuracy": max(utility_runs),
                "utility_runs": [
                    {"seed": seed, "fused_accuracy": accuracy}
                    for seed, accuracy in zip(args.utility_seeds, utility_runs)
                ],
            }
            candidate["utility_pass"] = (
                candidate["minimum_fused_accuracy"] > required_accuracy
            )
            if candidate["utility_pass"] or args.probe_all:
                training_runs = []
                inference_runs = []
                for seed in args.attack_seeds:
                    training_runs.append(
                        evaluate_attack(
                            reconstructor,
                            legacy.local,
                            semantic,
                            attack_training_loader,
                            device,
                            legacy_noise,
                            semantic_noise,
                            seed,
                        )["mse"]
                    )
                    inference_runs.append(
                        evaluate_attack(
                            reconstructor,
                            legacy.local,
                            semantic,
                            attack_inference_loader,
                            device,
                            legacy_noise,
                            semantic_noise,
                            seed + 100_000,
                        )["mse"]
                    )
                candidate.update(
                    decoder_training_mse_mean=float(np.mean(training_runs)),
                    decoder_training_mse_min=min(training_runs),
                    decoder_inference_mse_mean=float(np.mean(inference_runs)),
                    decoder_inference_mse_min=min(inference_runs),
                    attack_runs=[
                        {
                            "seed": seed,
                            "training_mse": training_mse,
                            "inference_mse": inference_mse,
                        }
                        for seed, training_mse, inference_mse in zip(
                            args.attack_seeds, training_runs, inference_runs
                        )
                    ],
                )
                candidate["minimum_normalised_privacy_margin"] = min(
                    candidate["decoder_training_mse_min"]
                    / args.training_mse_threshold,
                    candidate["decoder_inference_mse_min"]
                    / args.inference_mse_threshold,
                )
                candidate["probe_pass"] = (
                    candidate["decoder_training_mse_min"]
                    > args.training_mse_threshold
                    and candidate["decoder_inference_mse_min"]
                    > args.inference_mse_threshold
                )
            else:
                candidate["probe_pass"] = False
                candidate["minimum_normalised_privacy_margin"] = None
            candidates.append(candidate)
            print(json.dumps(candidate, sort_keys=True), flush=True)

    selected = select_probe_candidate(candidates)
    result = {
        "status": "SCREEN_PASS" if selected is not None else "SCREEN_FAIL",
        "screening_only": True,
        "formal_claim_requires_retraining_attackers": True,
        "required_accuracy": required_accuracy,
        "training_mse_threshold": args.training_mse_threshold,
        "inference_mse_threshold": args.inference_mse_threshold,
        "dual_path_checkpoint": str(args.dual_path_checkpoint),
        "attacker_checkpoint": str(args.attacker_checkpoint),
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
