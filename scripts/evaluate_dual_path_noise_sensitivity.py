#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--dual-path-checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        help="override the machine-specific legacy path stored in each checkpoint",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--legacy-noise", nargs="+", required=True, type=float)
    parser.add_argument("--semantic-noise", nargs="+", required=True, type=float)
    parser.add_argument(
        "--evaluation-seeds", nargs="+", type=int, default=[100126, 200126, 300126]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def aggregate(rows: list[dict]) -> list[dict]:
    points = []
    keys = sorted(
        {(row["legacy_noise_std"], row["semantic_noise_std"]) for row in rows}
    )
    for legacy_noise, semantic_noise in keys:
        selected = [
            row
            for row in rows
            if row["legacy_noise_std"] == legacy_noise
            and row["semantic_noise_std"] == semantic_noise
        ]
        values = np.asarray(
            [row["fused_accuracy"] for row in selected], dtype=np.float64
        )
        points.append(
            {
                "legacy_noise_std": legacy_noise,
                "semantic_noise_std": semantic_noise,
                "mean_fused_accuracy": float(values.mean()),
                "standard_deviation": float(
                    values.std(ddof=1) if values.size > 1 else 0.0
                ),
                "minimum_fused_accuracy": float(values.min()),
                "maximum_fused_accuracy": float(values.max()),
                "runs": int(values.size),
            }
        )
    return points


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, points: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Utility sensitivity to the deployment noise levels on FaceScrub. Reconstruction results at the selected operating points use attackers retrained at the corresponding noise level.}",
        r"\label{tab:noise_sensitivity_utility}",
        r"\begin{tabular}{ccc}",
        r"\toprule",
        r"$\sigma_g$ & $\sigma_s$ & Accuracy (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    for item in points:
        lines.append(
            f"{item['legacy_noise_std']:.2f} & {item['semantic_noise_std']:.2f} & "
            f"{100.0 * item['mean_fused_accuracy']:.2f} $\\pm$ "
            f"{100.0 * item['standard_deviation']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.legacy_noise + args.semantic_noise) < 0:
        raise ValueError("noise standard deviations must be non-negative")
    if len(args.legacy_noise) != len(args.semantic_noise):
        raise ValueError("legacy and semantic noise lists must define paired points")
    if not args.evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")
    points = list(dict.fromkeys(zip(args.legacy_noise, args.semantic_noise)))
    seed_everything(args.evaluation_seeds[0])
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint_path in args.dual_path_checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint["args"]
        _, validation_loader, class_to_index = build_loaders(
            args.data_root, args.batch_size, args.workers, args.evaluation_seeds[0]
        )
        if class_to_index != checkpoint["class_to_index"]:
            raise ValueError("checkpoint and validation class mappings differ")
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

        for legacy_noise, semantic_noise in points:
            for evaluation_seed in args.evaluation_seeds:
                metrics = evaluate(
                    legacy,
                    semantic,
                    fusion,
                    validation_loader,
                    device,
                    legacy_noise,
                    semantic_noise,
                    evaluation_seed,
                )
                rows.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "target_seed": int(checkpoint_args["seed"]),
                        "evaluation_seed": evaluation_seed,
                        "legacy_noise_std": legacy_noise,
                        "semantic_noise_std": semantic_noise,
                        **metrics,
                    }
                )

    summary = {
        "status": "PASS",
        "protocol": {
            "paired_noise_points": [
                {"legacy_noise_std": g, "semantic_noise_std": s} for g, s in points
            ],
            "target_checkpoints": [str(path) for path in args.dual_path_checkpoints],
            "evaluation_seeds": args.evaluation_seeds,
            "fresh_attackers_required_for_privacy_claims": True,
        },
        "aggregate": aggregate(rows),
        "runs": rows,
    }
    (args.output_dir / "noise_utility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "noise_utility_runs.csv", rows)
    write_latex(args.output_dir / "noise_utility_table.tex", summary["aggregate"])
    print(json.dumps({"status": "PASS", "runs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
