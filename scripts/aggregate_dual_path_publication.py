#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


PUBLISHED = {
    "accuracy": 0.8033,
    "decoder_training_mse": 0.0182,
    "decoder_inference_mse": 0.0211,
    "gan_training_mse": 0.0212,
    "gan_inference_mse": 0.0231,
}
PRIMARY_ATTACKS = ("decoder", "gan")
KNOWLEDGE_SETTINGS = ("training", "inference")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bootstrap_mean_ci(
    values: list[float], confidence: float = 0.95, draws: int = 20_000
) -> dict:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        lower = upper = float(array[0])
    else:
        generator = np.random.default_rng(20_260_804)
        indices = generator.integers(0, len(array), size=(draws, len(array)))
        means = array[indices].mean(axis=1)
        tail = (1.0 - confidence) / 2.0
        lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1) if len(array) > 1 else 0.0),
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "n_target_seeds": len(array),
    }


def target_attack_metrics(
    results_root: Path,
    target_seed: int,
    attacker_seeds: list[int],
    attack: str,
    knowledge: str,
) -> dict:
    records = []
    for attacker_seed in attacker_seeds:
        path = (
            results_root
            / "attacks"
            / f"target_seed{target_seed}"
            / f"attacker_seed{attacker_seed}"
            / f"{attack}_{knowledge}"
            / "attack_metrics.json"
        )
        payload = load_json(path)
        evaluation = payload["evaluation"]
        records.append(
            {
                "target_seed": target_seed,
                "attacker_seed": attacker_seed,
                "attack": attack,
                "knowledge": knowledge,
                "mse": float(evaluation["mse"]),
                "mae": float(evaluation["mae"]),
                "psnr": float(evaluation["psnr"]),
                "ssim": float(evaluation["ssim"]),
                "lpips": float(evaluation["lpips"]),
                "identity_cosine_similarity": float(evaluation["identity_cosine_similarity"]),
                "path": str(path),
            }
        )
    aggregate = {
        metric: float(np.mean([record[metric] for record in records]))
        for metric in ("mse", "mae", "psnr", "ssim")
    }
    return {"mean": aggregate, "attacker_runs": records}


def collect(results_root: Path, target_seeds: list[int], attacker_seeds: list[int]) -> dict:
    utility_records = []
    attack_records = []
    efficiency_records = []
    component_records = []
    stage1_records = []
    target_attack_means: dict[str, list[float]] = {}
    for target_seed in target_seeds:
        utility_path = (
            results_root / "targets" / f"seed{target_seed}" / "utility_l031_s010.json"
        )
        utility = load_json(utility_path)
        stage1_path = (
            results_root / "targets" / f"seed{target_seed}" / "utility_stage1_l031_s010.json"
        )
        stage1 = load_json(stage1_path)
        profile_path = (
            results_root / "targets" / f"seed{target_seed}" / "efficiency_profile.json"
        )
        profile = load_json(profile_path)
        utility_records.append(
            {
                "target_seed": target_seed,
                "accuracy": float(utility["mean_repeated_validation_accuracy"]),
                "conservative_accuracy": float(
                    utility["conservative_repeated_validation_accuracy"]
                ),
                "path": str(utility_path),
            }
        )
        stage1_records.append(
            {
                "target_seed": target_seed,
                "accuracy": float(stage1["mean_repeated_validation_accuracy"]),
                "conservative_accuracy": float(
                    stage1["conservative_repeated_validation_accuracy"]
                ),
                "path": str(stage1_path),
            }
        )
        for component in ("legacy", "semantic", "fused"):
            component_records.append(
                {
                    "target_seed": target_seed,
                    "component": component,
                    "accuracy": float(
                        np.mean(
                            [run[f"{component}_accuracy"] for run in utility["utility_runs"]]
                        )
                    ),
                }
            )
        efficiency_records.append({"target_seed": target_seed, **profile})
        for attack in (*PRIMARY_ATTACKS, "adaptive"):
            for knowledge in KNOWLEDGE_SETTINGS:
                if attack == "adaptive":
                    used_attacker_seeds = attacker_seeds[:1]
                else:
                    used_attacker_seeds = attacker_seeds
                result = target_attack_metrics(
                    results_root,
                    target_seed,
                    used_attacker_seeds,
                    attack,
                    knowledge,
                )
                key = f"{attack}_{knowledge}_mse"
                target_attack_means.setdefault(key, []).append(result["mean"]["mse"])
                attack_records.extend(result["attacker_runs"])
    return {
        "utility_records": utility_records,
        "attack_records": attack_records,
        "efficiency_records": efficiency_records,
        "component_records": component_records,
        "stage1_records": stage1_records,
        "target_attack_means": target_attack_means,
    }


def build_summary(
    results_root: Path, target_seeds: list[int], attacker_seeds: list[int]
) -> dict:
    collected = collect(results_root, target_seeds, attacker_seeds)
    accuracy_values = [row["accuracy"] for row in collected["utility_records"]]
    conservative_values = [row["conservative_accuracy"] for row in collected["utility_records"]]
    intervals = {
        "accuracy": bootstrap_mean_ci(accuracy_values),
        "conservative_accuracy": bootstrap_mean_ci(conservative_values),
    }
    for key, values in collected["target_attack_means"].items():
        intervals[key] = bootstrap_mean_ci(values)

    observed = {
        "accuracy": intervals["accuracy"]["mean"],
        **{
            f"{attack}_{knowledge}_mse": intervals[f"{attack}_{knowledge}_mse"]["mean"]
            for attack in PRIMARY_ATTACKS
            for knowledge in KNOWLEDGE_SETTINGS
        },
    }
    checks = {name: observed[name] > threshold for name, threshold in PUBLISHED.items()}
    confidence_checks = {
        name: intervals[name]["lower"] > threshold for name, threshold in PUBLISHED.items()
    }
    environment_path = results_root / "environment_gate.json"
    environment = load_json(environment_path) if environment_path.is_file() else {}
    analysis_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "strictly_dominates_published_mean": all(checks.values()),
        "confidence_supported_dominance": all(confidence_checks.values()),
        "checks": checks,
        "confidence_checks": confidence_checks,
        "observed": observed,
        "published": PUBLISHED,
        "intervals": intervals,
        "target_seeds": target_seeds,
        "attacker_seeds": attacker_seeds,
        "code_provenance": {
            "target_training_revision": environment.get("code_revision"),
            "analysis_revision": analysis_revision,
        },
        **collected,
        "comparison_source": {
            "title": (
                "Theoretical Insights in Model Inversion Robustness and Conditional "
                "Entropy Maximization for Collaborative Inference Systems"
            ),
            "venue": "CVPR 2025",
            "dataset": "FaceScrub",
            "table": 3,
            "baseline": "Noise_ARL+CEM",
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_ci(interval: dict, scale: float = 1.0) -> str:
    return (
        f"{interval['mean'] * scale:.4f} "
        f"[{interval['lower'] * scale:.4f}, {interval['upper'] * scale:.4f}]"
    )


def write_tables(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = summary["intervals"]
    rows = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Accuracy & Dec.-train MSE & Dec.-infer MSE & GAN-train MSE & GAN-infer MSE \\\\",
        "\\midrule",
        ("Noise\\_ARL+CEM & 0.8033 & 0.0182 & 0.0211 & 0.0212 & 0.0231 \\\\"),
        (
            "Dual-path CEM & "
            + " & ".join(
                format_mean_ci(intervals[key])
                for key in (
                    "accuracy",
                    "decoder_training_mse",
                    "decoder_inference_mse",
                    "gan_training_mse",
                    "gan_inference_mse",
                )
            )
            + " \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "results_table.tex").write_text("\n".join(rows) + "\n")

    attack_rows = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Attack & Knowledge & MSE & MAE & PSNR & SSIM & LPIPS & ID cosine \\\\",
        "\\midrule",
    ]
    for attack in (*PRIMARY_ATTACKS, "adaptive"):
        for knowledge in KNOWLEDGE_SETTINGS:
            selected = [
                row
                for row in summary["attack_records"]
                if row["attack"] == attack and row["knowledge"] == knowledge
            ]
            means = {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in (
                    "mse",
                    "mae",
                    "psnr",
                    "ssim",
                    "lpips",
                    "identity_cosine_similarity",
                )
            }
            attack_rows.append(
                f"{attack.title()} & {knowledge.title()} & "
                f"{means['mse']:.4f} & {means['mae']:.4f} & "
                f"{means['psnr']:.2f} & {means['ssim']:.4f} & "
                f"{means['lpips']:.4f} & "
                f"{means['identity_cosine_similarity']:.4f} \\\\"
            )
    attack_rows.extend(("\\bottomrule", "\\end{tabular}"))
    (output_dir / "attacker_table.tex").write_text("\n".join(attack_rows) + "\n")

    profiles = summary["efficiency_records"]
    efficiency_means = {
        key: float(np.mean([float(row[key]) for row in profiles]))
        for key in (
            "client_parameters",
            "server_parameters",
            "total_payload_elements",
            "transmitted_bytes_per_sample_fp32",
            "client_latency_ms_per_sample_mean",
            "server_latency_ms_per_sample_mean",
            "end_to_end_latency_ms_per_sample_mean",
        )
    }
    efficiency_rows = [
        "\\begin{tabular}{lrlr}",
        "\\toprule",
        "Metric & Value & Metric & Value \\\\",
        "\\midrule",
        "Client params & "
        f"{efficiency_means['client_parameters'] / 1e6:.2f}M & "
        "Server params & "
        f"{efficiency_means['server_parameters'] / 1e6:.2f}M \\\\",
        "Payload values & "
        f"{efficiency_means['total_payload_elements']:.0f} & "
        "Payload bytes & "
        f"{efficiency_means['transmitted_bytes_per_sample_fp32']:.0f} \\\\",
        "Client latency & "
        f"{efficiency_means['client_latency_ms_per_sample_mean']:.3f} ms & "
        "End-to-end & "
        f"{efficiency_means['end_to_end_latency_ms_per_sample_mean']:.3f} ms \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "efficiency_table.tex").write_text("\n".join(efficiency_rows) + "\n")

    component_rows = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Prediction path & Accuracy \\\\",
        "\\midrule",
    ]
    for component, label in (
        ("legacy", "Protected spatial path only"),
        ("semantic", "Semantic token only"),
        ("fused", "Calibrated fusion"),
    ):
        values = [
            row["accuracy"]
            for row in summary["component_records"]
            if row["component"] == component
        ]
        component_rows.append(f"{label} & {np.mean(values):.4f} \\\\ ")
    stage1_values = [row["accuracy"] for row in summary["stage1_records"]]
    component_rows.append(
        f"Fusion without robust server adaptation & {np.mean(stage1_values):.4f} \\\\"
    )
    component_rows.extend(("\\bottomrule", "\\end{tabular}"))
    (output_dir / "ablation_table.tex").write_text("\n".join(component_rows) + "\n")


def write_narratives(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = summary["intervals"]
    published = summary["published"]
    accuracy = intervals["accuracy"]
    accuracy_margin = 100.0 * (accuracy["mean"] - published["accuracy"])
    endpoint_labels = {
        "decoder_training_mse": "decoder training knowledge",
        "decoder_inference_mse": "decoder inference knowledge",
        "gan_training_mse": "GAN training knowledge",
        "gan_inference_mse": "GAN inference knowledge",
    }
    metric_labels = {"accuracy": "accuracy", **endpoint_labels}
    endpoint_text = "; ".join(
        f"{label}: {intervals[key]['mean']:.4f} "
        f"[{intervals[key]['lower']:.4f}, {intervals[key]['upper']:.4f}]"
        for key, label in endpoint_labels.items()
    )
    failed_means = [name for name, passed in summary["checks"].items() if not passed]
    failed_confidence = [
        name for name, passed in summary["confidence_checks"].items() if not passed
    ]
    if failed_means:
        gate_text = (
            "The mean did not exceed the pre-specified reference for "
            + ", ".join(metric_labels[name] for name in failed_means)
            + ", so the results do not establish improvement on every endpoint."
        )
    else:
        gate_text = "All five means were above the corresponding published reference."
    if failed_confidence:
        confidence_text = (
            "The lower 95\\% bootstrap bound did not clear the reference for "
            + ", ".join(metric_labels[name] for name in failed_confidence)
            + ". The observed means improved, but this direction was not retained "
            "by every lower confidence bound."
        )
    else:
        confidence_text = (
            "The lower 95\\% bootstrap bound also remained above the reference "
            "for every endpoint."
        )
    main_text = (
        f"Mean top-1 accuracy across the five target models was "
        f"{100.0 * accuracy['mean']:.2f}\\% "
        f"[{100.0 * accuracy['lower']:.2f}\\%, "
        f"{100.0 * accuracy['upper']:.2f}\\%], a gain of "
        f"{accuracy_margin:.2f} percentage points over the published "
        f"Noise\\_ARL+\\allowbreak CEM result. The four reconstruction endpoints "
        f"were {endpoint_text}. "
        f"{gate_text}\n\n{confidence_text}\n"
    )
    (output_dir / "main_results_narrative.tex").write_text(main_text)

    if failed_means:
        abstract_text = (
            f"Mean top-1 accuracy over five target seeds was "
            f"{100.0 * accuracy['mean']:.2f}\\%. At least one pre-specified "
            "accuracy or MSE reference was not exceeded.\n"
        )
        conclusion_text = (
            "At least one five-target mean remained below its pre-specified "
            "reference. The component and stronger-attacker results show where "
            "the dual-path design helps, but the experiment does not establish "
            "an improvement on every privacy and utility endpoint.\n"
        )
    else:
        abstract_text = (
            f"Mean top-1 accuracy over five target seeds was "
            f"{100.0 * accuracy['mean']:.2f}\\%. Decoder MSE was "
            f"{intervals['decoder_training_mse']['mean']:.4f} under training "
            f"knowledge and {intervals['decoder_inference_mse']['mean']:.4f} "
            f"under inference knowledge; GAN MSE was "
            f"{intervals['gan_training_mse']['mean']:.4f} and "
            f"{intervals['gan_inference_mse']['mean']:.4f}, respectively. Each "
            "mean exceeded the corresponding published Noise\\_ARL+CEM value.\n"
        )
        if failed_confidence:
            conclusion_text = (
                f"Accuracy reached {100.0 * accuracy['mean']:.2f}\\%, and the "
                "four decoder and GAN MSE means exceeded their published values. "
                "Some lower bootstrap bounds remained below their references, "
                "so the evidence is limited to the observed means. The second "
                "client path adds measured cost, and the experiment covers only "
                "FaceScrub.\n"
            )
        else:
            conclusion_text = (
                f"Accuracy reached {100.0 * accuracy['mean']:.2f}\\%, and every "
                "decoder and GAN MSE mean exceeded its published Noise\\_ARL+CEM "
                "reference. The same was true of all lower 95\\% bootstrap bounds. "
                "Under the stated FaceScrub threat model, the method therefore "
                "improved both measured utility and reconstruction resistance. "
                "This gain includes the measured cost of the second client path "
                "and has not yet been established on another dataset.\n"
            )
    (output_dir / "abstract_results.tex").write_text(abstract_text)
    (output_dir / "conclusion_results.tex").write_text(conclusion_text)

    def attack_mean(attack: str, knowledge: str, metric: str) -> float:
        values = [
            row[metric]
            for row in summary["attack_records"]
            if row["attack"] == attack and row["knowledge"] == knowledge
        ]
        return float(np.mean(values))

    attack_text = (
        "Under inference knowledge, MSE was "
        f"{attack_mean('decoder', 'inference', 'mse'):.4f} for the residual "
        f"decoder and {attack_mean('gan', 'inference', 'mse'):.4f} for the GAN. "
        "The wider noise-averaged decoder obtained "
        f"{attack_mean('adaptive', 'inference', 'mse'):.4f}, with LPIPS "
        f"{attack_mean('adaptive', 'inference', 'lpips'):.4f}, SSIM "
        f"{attack_mean('adaptive', 'inference', 'ssim'):.4f}, and identity "
        "cosine similarity "
        f"{attack_mean('adaptive', 'inference', 'identity_cosine_similarity'):.4f}. "
        "We report the perceptual and identity measures because pixel error alone "
        "does not show whether a reconstruction still identifies the subject.\n"
    )
    (output_dir / "attack_results_narrative.tex").write_text(attack_text)

    component_means = {
        component: float(
            np.mean(
                [
                    row["accuracy"]
                    for row in summary["component_records"]
                    if row["component"] == component
                ]
            )
        )
        for component in ("legacy", "semantic", "fused")
    }
    stage1_mean = float(np.mean([row["accuracy"] for row in summary["stage1_records"]]))
    ablation_text = (
        "At deployment noise, accuracy was "
        f"{100.0 * component_means['legacy']:.2f}\\% with only the protected "
        f"spatial path and {100.0 * component_means['semantic']:.2f}\\% with "
        f"only the semantic path. Their calibrated fusion reached "
        f"{100.0 * component_means['fused']:.2f}\\%. Before robust server "
        f"adaptation, the same fusion reached {100.0 * stage1_mean:.2f}\\%, "
        "showing the contribution of Stage 2 at the deployment noise.\n"
    )
    (output_dir / "ablation_results_narrative.tex").write_text(ablation_text)

    profiles = summary["efficiency_records"]
    profile_mean = {
        key: float(np.mean([float(row[key]) for row in profiles]))
        for key in (
            "client_parameters",
            "total_payload_elements",
            "transmitted_bytes_per_sample_fp32",
            "client_latency_ms_per_sample_mean",
            "end_to_end_latency_ms_per_sample_mean",
        )
    }
    efficiency_text = (
        f"The client has {profile_mean['client_parameters'] / 1e6:.2f} million "
        "parameters and sends "
        f"{profile_mean['total_payload_elements']:.0f} FP32 values "
        f"({profile_mean['transmitted_bytes_per_sample_fp32']:.0f} bytes) per "
        f"image. Mean client and end-to-end latency on the experiment GPU were "
        f"{profile_mean['client_latency_ms_per_sample_mean']:.3f} ms and "
        f"{profile_mean['end_to_end_latency_ms_per_sample_mean']:.3f} ms per "
        "image, respectively. These GPU measurements quantify the added cost but "
        "should not be interpreted as mobile-device latency.\n"
    )
    (output_dir / "efficiency_results_narrative.tex").write_text(efficiency_text)


def write_privacy_utility_plot(summary: dict, output_dir: Path) -> None:
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    accuracies = [row["accuracy"] for row in summary["utility_records"]]
    mse_values = summary["target_attack_means"]["decoder_inference_mse"]
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    axis.scatter(
        [PUBLISHED["accuracy"]],
        [PUBLISHED["decoder_inference_mse"]],
        marker="s",
        color="#6b7280",
        s=60,
        label="Published Noise_ARL+CEM",
    )
    axis.scatter(
        accuracies,
        mse_values,
        color="#0f766e",
        s=45,
        label="Dual-path CEM target seeds",
    )
    axis.scatter(
        [np.mean(accuracies)],
        [np.mean(mse_values)],
        marker="*",
        color="#b91c1c",
        s=120,
        label="Dual-path CEM mean",
    )
    axis.set_xlabel("Top-1 accuracy")
    axis.set_ylabel("Decoder inference MSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "privacy_utility.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "privacy_utility.png", dpi=400, bbox_inches="tight")
    plt.close(figure)


def write_architecture_diagram(output_dir: Path) -> None:
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(10.2, 4.2))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 4.2)
    axis.axis("off")

    def box(x, y, width, height, text, colour):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.1,
            edgecolor="#1f2937",
            facecolor=colour,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, colour="#374151"):
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=colour,
            )
        )

    axis.text(2.0, 4.02, "Client", ha="center", fontsize=11, fontweight="bold")
    axis.text(9.4, 4.08, "Server", ha="center", fontsize=11, fontweight="bold")
    axis.axvline(4.75, color="#6b7280", linestyle="--", linewidth=1.0)
    box(0.15, 1.65, 1.0, 0.7, "Input\nimage", "#f3f4f6")
    box(1.55, 2.55, 1.75, 0.75, "Frozen Slot-CEM\nspatial encoder", "#dbeafe")
    box(1.55, 0.70, 1.75, 0.75, "MobileNetV3 + GAP\n256-D projection", "#dcfce7")
    box(3.65, 2.55, 0.85, 0.75, "Gaussian\nnoise", "#fef3c7")
    box(3.65, 0.70, 0.85, 0.75, "Gaussian\nnoise", "#fef3c7")
    box(5.05, 2.55, 1.65, 0.75, "VGG cloud +\nspatial classifier", "#dbeafe")
    box(5.05, 0.70, 1.65, 0.75, "Semantic\nclassifier", "#dcfce7")
    box(7.15, 1.65, 1.25, 0.75, "Temperature-\ncalibrated fusion", "#ede9fe")
    box(8.85, 1.65, 1.1, 0.75, "Identity\nlogits", "#f3f4f6")
    box(5.25, 3.45, 1.9, 0.55, "Joint two-path attacker", "#fee2e2")
    box(7.65, 3.45, 1.45, 0.55, "Reconstruction", "#fee2e2")

    arrow(1.15, 2.0, 1.55, 2.92)
    arrow(1.15, 2.0, 1.55, 1.08)
    arrow(3.30, 2.92, 3.65, 2.92)
    arrow(3.30, 1.08, 3.65, 1.08)
    arrow(4.50, 2.92, 5.05, 2.92)
    arrow(4.50, 1.08, 5.05, 1.08)
    arrow(6.70, 2.92, 7.28, 2.36)
    arrow(6.70, 1.08, 7.28, 1.69)
    arrow(8.40, 2.02, 8.85, 2.02)
    arrow(4.25, 3.30, 5.40, 3.52, "#b91c1c")
    arrow(4.25, 1.45, 5.85, 3.45, "#b91c1c")
    arrow(7.15, 3.72, 7.65, 3.72, "#b91c1c")

    axis.text(4.70, 0.18, "client--server boundary", ha="right", fontsize=8, color="#4b5563")
    figure.tight_layout(pad=0.4)
    figure.savefig(output_dir / "architecture.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "architecture.png", dpi=400, bbox_inches="tight")
    plt.close(figure)


def write_manifest(paths: list[Path], output: Path) -> None:
    files = {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
        if path.is_file()
    }
    save_json(output, {"schema_version": 1, "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--paper-output", required=True, type=Path)
    parser.add_argument("--target-seeds", nargs="+", type=int, default=list(range(126, 131)))
    parser.add_argument("--attacker-seeds", nargs="+", type=int, default=[10125, 20125, 30125])
    args = parser.parse_args()
    summary = build_summary(args.results_root, args.target_seeds, args.attacker_seeds)
    summary_path = args.results_root / "formal_summary.json"
    save_json(summary_path, summary)
    utility_csv = args.results_root / "utility_target_level.csv"
    attacks_csv = args.results_root / "attack_runs.csv"
    write_csv(utility_csv, summary["utility_records"])
    write_csv(attacks_csv, summary["attack_records"])
    write_tables(summary, args.paper_output)
    write_narratives(summary, args.paper_output)
    from generate_publication_figures import generate_all

    qualitative_source = args.paper_output / "qualitative_grid.png"
    generate_all(
        summary,
        args.paper_output,
        qualitative_source if qualitative_source.is_file() else None,
    )
    generated = [
        summary_path,
        utility_csv,
        attacks_csv,
        *args.paper_output.glob("*"),
    ]
    write_manifest(generated, args.results_root / "publication_materials_manifest.json")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "confidence_supported_dominance": summary["confidence_supported_dominance"],
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
