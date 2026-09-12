#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    legacy_logits,
    load_legacy,
    restore_legacy_server,
    seed_everything,
)


STRATEGIES = (
    "spatial_only",
    "semantic_only",
    "average_logits",
    "fixed_weight_035",
    "learned_weight_no_temperature",
    "confidence_gating",
    "calibrated_learned_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-path-checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        help="override the machine-specific legacy path stored in each checkpoint",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--legacy-noise-std", type=float, default=0.31)
    parser.add_argument("--semantic-noise-std", type=float, default=0.10)
    parser.add_argument(
        "--evaluation-seeds", nargs="+", type=int, default=[100126, 200126, 300126]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fuse_logits(
    spatial_logits: torch.Tensor,
    semantic_logits: torch.Tensor,
    fusion: CalibratedLogitFusion,
    strategy: str,
) -> torch.Tensor:
    if spatial_logits.shape != semantic_logits.shape:
        raise ValueError("spatial and semantic logits must have the same shape")
    if strategy == "spatial_only":
        return spatial_logits
    if strategy == "semantic_only":
        return semantic_logits
    if strategy == "average_logits":
        return 0.5 * spatial_logits + 0.5 * semantic_logits
    if strategy == "fixed_weight_035":
        return 0.65 * spatial_logits + 0.35 * semantic_logits
    if strategy == "learned_weight_no_temperature":
        weight = fusion.mix_logit.sigmoid()
        return (1.0 - weight) * spatial_logits + weight * semantic_logits
    if strategy == "confidence_gating":
        spatial_confidence = spatial_logits.softmax(dim=1).amax(dim=1, keepdim=True)
        semantic_confidence = semantic_logits.softmax(dim=1).amax(dim=1, keepdim=True)
        return torch.where(
            spatial_confidence >= semantic_confidence,
            spatial_logits,
            semantic_logits,
        )
    if strategy == "calibrated_learned_fusion":
        return fusion(spatial_logits, semantic_logits)
    raise ValueError(f"unsupported fusion strategy: {strategy}")


@torch.no_grad()
def evaluate_strategies(
    legacy,
    semantic: GlobalSemanticBottleneck,
    fusion: CalibratedLogitFusion,
    loader,
    device: torch.device,
    legacy_noise_std: float,
    semantic_noise_std: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    legacy.eval()
    semantic.eval()
    fusion.eval()
    legacy_generator = torch.Generator(device=device).manual_seed(seed)
    semantic_generator = torch.Generator(device=device).manual_seed(seed + 1)
    totals = {
        strategy: {"correct": 0, "loss": 0.0, "count": 0}
        for strategy in STRATEGIES
    }
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spatial_logits, _ = legacy_logits(
            legacy, images, legacy_noise_std, legacy_generator
        )
        semantic_logits, _ = semantic(
            images, semantic_noise_std, semantic_generator
        )
        for strategy in STRATEGIES:
            logits = fuse_logits(spatial_logits, semantic_logits, fusion, strategy)
            batch = images.shape[0]
            totals[strategy]["correct"] += int((logits.argmax(dim=1) == labels).sum())
            totals[strategy]["loss"] += float(F.cross_entropy(logits, labels)) * batch
            totals[strategy]["count"] += batch
    return {
        strategy: {
            "accuracy": values["correct"] / values["count"],
            "cross_entropy": values["loss"] / values["count"],
            "count": values["count"],
        }
        for strategy, values in totals.items()
    }


def aggregate(rows: list[dict]) -> dict:
    grouped = {}
    for strategy in STRATEGIES:
        values = np.asarray(
            [row["accuracy"] for row in rows if row["strategy"] == strategy],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError(f"missing fusion results for {strategy}")
        grouped[strategy] = {
            "mean_accuracy": float(values.mean()),
            "standard_deviation": float(
                values.std(ddof=1) if values.size > 1 else 0.0
            ),
            "minimum_accuracy": float(values.min()),
            "maximum_accuracy": float(values.max()),
            "runs": int(values.size),
        }
    return grouped


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_name(strategy: str) -> str:
    return {
        "spatial_only": "G-Path only",
        "semantic_only": "S-Path only",
        "average_logits": "Equal logit average",
        "fixed_weight_035": "Fixed 0.35 semantic weight",
        "learned_weight_no_temperature": "Learned weight, no calibration",
        "confidence_gating": "Per-sample confidence gate",
        "calibrated_learned_fusion": "Calibrated learned fusion",
    }[strategy]


def write_latex(path: Path, summary: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fusion ablation on FaceScrub at the fixed deployment noise. The released representations are identical in every row; only the server-side combination rule changes.}",
        r"\label{tab:fusion_ablation}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Fusion rule & Accuracy (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    for strategy in STRATEGIES:
        item = summary[strategy]
        lines.append(
            f"{display_name(strategy)} & "
            f"{100.0 * item['mean_accuracy']:.2f} $\\pm$ "
            f"{100.0 * item['standard_deviation']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.legacy_noise_std, args.semantic_noise_std) < 0:
        raise ValueError("noise standard deviations must be non-negative")
    if not args.evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")
    seed_everything(args.evaluation_seeds[0])
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    learned_parameters = []
    expected_class_mapping = None
    for checkpoint_path in args.dual_path_checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint["args"]
        _, validation_loader, class_to_index = build_loaders(
            args.data_root, args.batch_size, args.workers, args.evaluation_seeds[0]
        )
        if class_to_index != checkpoint["class_to_index"]:
            raise ValueError("checkpoint and validation class mappings differ")
        if expected_class_mapping is None:
            expected_class_mapping = class_to_index
        elif class_to_index != expected_class_mapping:
            raise ValueError("target checkpoints use different class mappings")

        legacy_dir = args.legacy_checkpoint_dir or Path(
            checkpoint["legacy_checkpoint_dir"]
        )
        legacy = load_legacy(legacy_dir, device)
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
        learned_parameters.append(
            {
                "checkpoint": str(checkpoint_path),
                "target_seed": int(checkpoint_args["seed"]),
                "semantic_weight": float(fusion.mix_logit.sigmoid().detach().cpu()),
                "spatial_temperature": float(
                    fusion.legacy_log_temperature.exp().clamp(0.25, 4.0).detach().cpu()
                ),
                "semantic_temperature": float(
                    fusion.semantic_log_temperature.exp().clamp(0.25, 4.0).detach().cpu()
                ),
            }
        )
        for evaluation_seed in args.evaluation_seeds:
            metrics = evaluate_strategies(
                legacy,
                semantic,
                fusion,
                validation_loader,
                device,
                args.legacy_noise_std,
                args.semantic_noise_std,
                evaluation_seed,
            )
            for strategy, values in metrics.items():
                rows.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "target_seed": int(checkpoint_args["seed"]),
                        "evaluation_seed": evaluation_seed,
                        "strategy": strategy,
                        **values,
                    }
                )

    result = {
        "status": "PASS",
        "protocol": {
            "target_checkpoints": [str(path) for path in args.dual_path_checkpoints],
            "evaluation_seeds": args.evaluation_seeds,
            "legacy_noise_std": args.legacy_noise_std,
            "semantic_noise_std": args.semantic_noise_std,
            "privacy_release_is_fixed_across_strategies": True,
        },
        "learned_fusion_parameters": learned_parameters,
        "aggregate": aggregate(rows),
        "runs": rows,
    }
    (args.output_dir / "fusion_ablation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "fusion_ablation_runs.csv", rows)
    write_latex(args.output_dir / "fusion_ablation_table.tex", result["aggregate"])
    print(json.dumps({"status": "PASS", "runs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
