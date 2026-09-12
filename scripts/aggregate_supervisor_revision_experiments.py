#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sample")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1) if array.size > 1 else 0.0),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "n": int(array.size),
    }


def collect_token_results(root: Path, dimensions: list[int], seeds: list[int]) -> dict:
    summary = {}
    for dimension in dimensions:
        rows = []
        for seed in seeds:
            payload = load_json(
                root
                / "token_dimension"
                / f"dim{dimension}"
                / f"seed{seed}"
                / "utility.json"
            )
            rows.append(
                {
                    "token_dimension": dimension,
                    "target_seed": seed,
                    "mean_accuracy": float(payload["mean_repeated_validation_accuracy"]),
                    "minimum_accuracy": float(
                        payload["conservative_repeated_validation_accuracy"]
                    ),
                }
            )
        summary[str(dimension)] = {
            "mean_accuracy": mean_std([row["mean_accuracy"] for row in rows]),
            "minimum_accuracy": mean_std([row["minimum_accuracy"] for row in rows]),
            "runs": rows,
        }
    return summary


def collect_noise_attacks(
    root: Path,
    points: list[tuple[float, float]],
    target_seeds: list[int],
    attacker_seeds: list[int],
    knowledge_settings: list[str],
) -> dict:
    summary = {}
    for legacy_noise, semantic_noise in points:
        point_name = f"g{legacy_noise:.2f}_s{semantic_noise:.2f}".replace(".", "")
        point_rows = []
        for target_seed in target_seeds:
            for attacker_seed in attacker_seeds:
                for knowledge in knowledge_settings:
                    payload = load_json(
                        root
                        / "noise_attacks"
                        / point_name
                        / f"target_seed{target_seed}"
                        / f"attacker_seed{attacker_seed}"
                        / f"decoder_{knowledge}"
                        / "attack_metrics.json"
                    )
                    point_rows.append(
                        {
                            "target_seed": target_seed,
                            "attacker_seed": attacker_seed,
                            "knowledge": knowledge,
                            **{
                                key: float(value)
                                for key, value in payload["evaluation"].items()
                                if isinstance(value, (int, float))
                            },
                        }
                    )
        by_knowledge = {}
        for knowledge in knowledge_settings:
            selected = [row for row in point_rows if row["knowledge"] == knowledge]
            by_knowledge[knowledge] = {
                metric: mean_std([row[metric] for row in selected])
                for metric in ("mse", "psnr", "ssim", "lpips", "identity_cosine_similarity")
            }
        summary[point_name] = {
            "legacy_noise_std": legacy_noise,
            "semantic_noise_std": semantic_noise,
            "aggregate": by_knowledge,
            "runs": point_rows,
        }
    return summary


def write_token_table(path: Path, token_summary: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity to the semantic token dimension on FaceScrub. All variants follow the same two-stage training protocol.}",
        r"\label{tab:token_dimension}",
        r"\begin{tabular}{cc}",
        r"\toprule",
        r"Token dimension & Accuracy (\%) $\uparrow$ \\",
        r"\midrule",
    ]
    for dimension in sorted(token_summary, key=int):
        stats = token_summary[dimension]["mean_accuracy"]
        lines.append(
            f"{dimension} & {100.0 * stats['mean']:.2f} $\\pm$ "
            f"{100.0 * stats['standard_deviation']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_noise_table(path: Path, noise_summary: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Decoder inversion after retraining the attacker at each noise operating point. Higher MSE and lower SSIM indicate stronger reconstruction resistance.}",
        r"\label{tab:noise_sensitivity_privacy}",
        r"\begin{tabular}{cclcc}",
        r"\toprule",
        r"$\sigma_g$ & $\sigma_s$ & Knowledge & MSE $\uparrow$ & SSIM $\downarrow$ \\",
        r"\midrule",
    ]
    for point in noise_summary.values():
        for index, knowledge in enumerate(("training", "inference")):
            metrics = point["aggregate"][knowledge]
            g = f"{point['legacy_noise_std']:.2f}" if index == 0 else ""
            s = f"{point['semantic_noise_std']:.2f}" if index == 0 else ""
            lines.append(
                f"{g} & {s} & {knowledge.title()} & "
                f"{metrics['mse']['mean']:.4f} & {metrics['ssim']['mean']:.4f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--token-dimensions", nargs="+", type=int, required=True)
    parser.add_argument("--token-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--noise-points", nargs="+", required=True)
    parser.add_argument("--noise-target-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--noise-attacker-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--knowledge", nargs="+", required=True)
    args = parser.parse_args()

    points = []
    for value in args.noise_points:
        legacy, semantic = value.split(",", maxsplit=1)
        points.append((float(legacy), float(semantic)))
    fusion = load_json(args.results_root / "fusion" / "fusion_ablation_summary.json")
    noise_utility = load_json(
        args.results_root / "noise_utility" / "noise_utility_summary.json"
    )
    identity = load_json(
        args.results_root
        / "identity_conditioned_attack"
        / "identity_conditioned_attack_summary.json"
    )
    token = collect_token_results(
        args.results_root, args.token_dimensions, args.token_seeds
    )
    noise_attack = collect_noise_attacks(
        args.results_root,
        points,
        args.noise_target_seeds,
        args.noise_attacker_seeds,
        args.knowledge,
    )
    result = {
        "status": "PASS",
        "fusion": fusion,
        "noise_utility": noise_utility,
        "identity_conditioned_attack": identity,
        "token_dimension": token,
        "noise_attack": noise_attack,
    }
    (args.results_root / "supervisor_revision_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_token_table(args.results_root / "token_dimension_table.tex", token)
    write_noise_table(args.results_root / "noise_attack_table.tex", noise_attack)
    print(json.dumps({"status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
