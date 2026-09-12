#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "identity_cosine_similarity",
    "semantic_identity_top1_accuracy",
    "reconstructed_identity_top1_accuracy",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bootstrap_mean_ci(values: list[float], draws: int = 20_000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot aggregate an empty sample")
    if array.size == 1:
        lower = upper = float(array[0])
    else:
        generator = np.random.default_rng(20_260_825)
        indices = generator.integers(0, array.size, size=(draws, array.size))
        means = array[indices].mean(axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1) if array.size > 1 else 0.0),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n": int(array.size),
    }


def collect(
    results_root: Path,
    target_seeds: list[int],
    attacker_seeds: list[int],
    knowledge_settings: list[str],
    scenarios: list[str],
) -> list[dict]:
    rows = []
    for target_seed in target_seeds:
        for attacker_seed in attacker_seeds:
            for knowledge in knowledge_settings:
                for scenario in scenarios:
                    path = (
                        results_root
                        / f"target_seed{target_seed}"
                        / f"attacker_seed{attacker_seed}"
                        / f"{scenario}_{knowledge}"
                        / "attack_metrics.json"
                    )
                    payload = load_json(path)
                    evaluation = payload["evaluation"]
                    rows.append(
                        {
                            "target_seed": target_seed,
                            "attacker_seed": attacker_seed,
                            "knowledge": knowledge,
                            "scenario": scenario,
                            "attack_view": payload["attack_view"],
                            "identity_conditioning": payload["identity_conditioning"],
                            **{metric: float(evaluation[metric]) for metric in METRICS},
                            "path": str(path),
                        }
                    )
    return rows


def summarise(rows: list[dict], knowledge_settings: list[str], scenarios: list[str]) -> dict:
    grouped = {}
    paired = {}
    for knowledge in knowledge_settings:
        grouped[knowledge] = {}
        knowledge_rows = [row for row in rows if row["knowledge"] == knowledge]
        for scenario in scenarios:
            scenario_rows = [row for row in knowledge_rows if row["scenario"] == scenario]
            grouped[knowledge][scenario] = {
                metric: bootstrap_mean_ci([row[metric] for row in scenario_rows])
                for metric in METRICS
            }

        baseline = {
            (row["target_seed"], row["attacker_seed"]): row
            for row in knowledge_rows
            if row["scenario"] == "joint_unconditioned"
        }
        paired[knowledge] = {}
        for scenario in scenarios:
            if scenario == "joint_unconditioned" or not baseline:
                continue
            scenario_rows = [row for row in knowledge_rows if row["scenario"] == scenario]
            paired[knowledge][scenario] = {}
            for metric in METRICS:
                differences = [
                    row[metric] - baseline[(row["target_seed"], row["attacker_seed"])][metric]
                    for row in scenario_rows
                ]
                paired[knowledge][scenario][metric] = bootstrap_mean_ci(differences)
    return {"aggregate": grouped, "paired_delta_vs_joint_unconditioned": paired}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_name(scenario: str) -> str:
    return {
        "g_only": "G-Path only",
        "joint_unconditioned": "Joint, no identity prior",
        "joint_predicted_identity": "Joint + predicted identity",
        "joint_oracle_identity": "Joint + oracle identity",
    }.get(scenario, scenario.replace("_", " ").title())


def write_latex_table(path: Path, summary: dict, scenarios: list[str]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Identity-conditioned joint inversion on FaceScrub. Lower MSE and LPIPS and higher SSIM and identity similarity indicate a stronger attacker. Oracle identity is an upper-bound diagnostic rather than information available to the deployed attacker.}",
        r"\label{tab:identity_conditioned_attack}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Knowledge & Attack input & MSE $\downarrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & ID cosine $\uparrow$ \\",
        r"\midrule",
    ]
    for knowledge, groups in summary["aggregate"].items():
        for index, scenario in enumerate(scenarios):
            metrics = groups[scenario]
            knowledge_label = knowledge.title() if index == 0 else ""
            lines.append(
                f"{knowledge_label} & {display_name(scenario)} & "
                f"{metrics['mse']['mean']:.4f} & "
                f"{metrics['ssim']['mean']:.4f} & "
                f"{metrics['lpips']['mean']:.4f} & "
                f"{metrics['identity_cosine_similarity']['mean']:.4f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--target-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--attacker-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--knowledge", nargs="+", choices=("training", "inference"), required=True
    )
    parser.add_argument("--scenarios", nargs="+", required=True)
    args = parser.parse_args()

    rows = collect(
        args.results_root,
        args.target_seeds,
        args.attacker_seeds,
        args.knowledge,
        args.scenarios,
    )
    summary = {
        "status": "PASS",
        "interpretation": {
            "predicted_identity": (
                "Compare the paired delta against joint_unconditioned. A negative "
                "MSE delta or positive SSIM/identity-similarity delta means that the "
                "deployable identity-conditioned attacker is stronger."
            ),
            "oracle_identity": (
                "This row is a worst-case upper bound and must not be presented as "
                "information exposed by the dual-path system."
            ),
        },
        "target_seeds": args.target_seeds,
        "attacker_seeds": args.attacker_seeds,
        "knowledge_settings": args.knowledge,
        "scenarios": args.scenarios,
        "runs": len(rows),
        **summarise(rows, args.knowledge, args.scenarios),
    }
    save_json(args.results_root / "identity_conditioned_attack_summary.json", summary)
    write_csv(args.results_root / "identity_conditioned_attack_runs.csv", rows)
    write_latex_table(
        args.results_root / "identity_conditioned_attack_table.tex",
        summary,
        args.scenarios,
    )
    print(json.dumps({"status": "PASS", "runs": len(rows)}))


if __name__ == "__main__":
    main()
