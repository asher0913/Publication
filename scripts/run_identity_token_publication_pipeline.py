#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "identity_token_publication"
GENERATED_CONFIGS = ROOT / "generated_configs" / "identity_token_publication"
LOG_ROOT = ROOT / "run_logs" / "identity_token_publication"
STATE_PATH = ROOT / "run_state" / "identity_token_publication.json"
PAPER_ROOT = ROOT / "paper" / "generated" / "identity_token"
EXPECTED_DATASET = {"classes": 526, "train": 39_969, "val": 4_170, "size": 48}
PUBLISHED_THRESHOLDS = ROOT / "configs" / "published_cem_facescrub_thresholds.json"
PUBLISHED_BASELINE = "noise_arl_cem"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"schema_version": 1, "started_at_utc": now(), "jobs": {}}


def is_under(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def reset_incomplete(paths: list[Path]) -> None:
    allowed = (RESULTS_ROOT, GENERATED_CONFIGS, PAPER_ROOT)
    for path in paths:
        if not any(is_under(path, root) for root in allowed):
            raise ValueError(f"refusing to reset path outside campaign roots: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def completion_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
    return True


def run_job(
    state: dict,
    name: str,
    command: list[str],
    completion: Path,
    reset_paths: list[Path],
    attempts: int,
    retry_seconds: float,
) -> None:
    if completion_valid(completion):
        state["jobs"][name] = {
            "status": "PASS",
            "completion": str(completion.relative_to(ROOT)),
            "recovered_from_artifact": True,
        }
        save_json(STATE_PATH, state)
        print(json.dumps({"job": name, "status": "SKIP_COMPLETE"}), flush=True)
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        reset_incomplete(reset_paths)
        log_path = LOG_ROOT / f"{name}.attempt{attempt}.log"
        state["jobs"][name] = {
            "status": "RUNNING",
            "attempt": attempt,
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
            "started_at_utc": now(),
        }
        save_json(STATE_PATH, state)
        print(json.dumps({"job": name, "status": "RUNNING", "attempt": attempt}), flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
        if result.returncode == 0 and completion_valid(completion):
            state["jobs"][name].update(status="PASS", finished_at_utc=now())
            save_json(STATE_PATH, state)
            print(json.dumps({"job": name, "status": "PASS"}), flush=True)
            return
        state["jobs"][name].update(
            status="RETRY" if attempt < attempts else "FAIL",
            returncode=result.returncode,
            finished_at_utc=now(),
        )
        save_json(STATE_PATH, state)
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"job failed after {attempts} attempts: {name}")


def write_config(name: str, config: dict) -> Path:
    path = GENERATED_CONFIGS / f"{name}.json"
    save_json(path, config)
    return path


def configured(base: dict, name: str, seed: int, overrides: dict | None = None) -> tuple[dict, Path]:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["attack"]["seed"] = 10_000 + seed
    config["output_dir"] = str((RESULTS_ROOT / name).relative_to(ROOT))
    for dotted_key, value in (overrides or {}).items():
        current = config
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    return config, write_config(name, config)


def validate_environment(data_root: Path) -> dict:
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = {
        "classes": int(manifest.get("classes", -1)),
        "train": int(manifest.get("split_counts", {}).get("train", -1)),
        "val": int(manifest.get("split_counts", {}).get("val", -1)),
        "size": int(manifest.get("size", -1)),
    }
    if observed != EXPECTED_DATASET:
        raise ValueError(f"FaceScrub manifest mismatch: {observed} != {EXPECTED_DATASET}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    record = {
        "status": "PASS",
        "dataset": manifest,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "validated_at_utc": now(),
    }
    save_json(RESULTS_ROOT / "environment_gate.json", record)
    return record


def train_target(
    state: dict,
    name: str,
    config_path: Path,
    attempts: int,
    retry_seconds: float,
    selection_only: bool,
) -> Path:
    run_dir = RESULTS_ROOT / name
    completion = run_dir / (
        "target_selection_metrics.json" if selection_only else "target_test_metrics.json"
    )
    command = [sys.executable, "scripts/train_target.py", "--config", str(config_path)]
    if selection_only:
        command.append("--selection-only")
    run_job(
        state,
        f"target_{name}",
        command,
        completion,
        [run_dir],
        attempts,
        retry_seconds,
    )
    return run_dir


def train_attack(
    state: dict,
    run_name: str,
    checkpoint: Path,
    attack_type: str,
    attack_seed: int,
    epochs: int,
    attempts: int,
    retry_seconds: float,
    group: str = "attacks",
) -> Path:
    output_dir = checkpoint.parent / group / attack_type / f"seed{attack_seed}"
    command = [
        sys.executable,
        "scripts/train_attack.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--attack-type",
        attack_type,
        "--seed",
        str(attack_seed),
        "--epochs",
        str(epochs),
        "--learning-rate",
        "0.005",
    ]
    run_job(
        state,
        f"attack_{run_name}_{attack_type}_{attack_seed}",
        command,
        output_dir / "attack_metrics.json",
        [output_dir],
        attempts,
        retry_seconds,
    )
    return output_dir / "attack_metrics.json"


def accuracy(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["best_validation_accuracy"])


def test_accuracy(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["test_accuracy"])


def formal_published_gate(target_csv: Path) -> dict:
    thresholds = json.loads(PUBLISHED_THRESHOLDS.read_text(encoding="utf-8"))
    baseline = thresholds["baselines"][PUBLISHED_BASELINE]
    accuracies = [
        test_accuracy(
            RESULTS_ROOT
            / "headline"
            / f"proposed_seed{seed}"
            / "target_test_metrics.json"
        )
        for seed in range(125, 130)
    ]
    with target_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    decoder_mse = [
        float(row["mse"])
        for row in rows
        if row["method"] == "identity_token_cem"
        and row["attack_type"] == "residual_decoder"
    ]
    gan_mse = [
        float(row["mse"])
        for row in rows
        if row["method"] == "identity_token_cem" and row["attack_type"] == "gan"
    ]
    if len(accuracies) != 5 or len(decoder_mse) != 5 or len(gan_mse) != 5:
        raise ValueError("formal gate requires five proposed target seeds per metric")
    observed = {
        "accuracy": mean(accuracies),
        "decoder_inference_mse": mean(decoder_mse),
        "gan_inference_mse": mean(gan_mse),
    }
    checks = {name: value > float(baseline[name]) for name, value in observed.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "strictly_dominates_published_mean": all(checks.values()),
        "observed_means": observed,
        "published_baseline": baseline,
        "checks": checks,
        "target_seed_accuracies": accuracies,
        "target_level_decoder_mse": decoder_mse,
        "target_level_gan_mse": gan_mse,
        "source": thresholds["source"],
        "compared_baseline": PUBLISHED_BASELINE,
    }
    save_json(RESULTS_ROOT / "formal_published_pareto_gate.json", result)
    return result


def run_smoke(state: dict, base: dict, attempts: int, retry_seconds: float) -> None:
    config, path = configured(
        base,
        "smoke",
        118,
        {
            "training.epochs": 1,
            "data.workers": 2,
            "model.pretrained": False,
        },
    )
    run_dir = train_target(state, "smoke", path, attempts, retry_seconds, True)
    train_attack(
        state,
        "smoke",
        run_dir / "checkpoint_best.pt",
        "residual_decoder",
        10_118,
        1,
        attempts,
        retry_seconds,
        group="smoke_attacks",
    )


def run_pilot(state: dict, base: dict, attempts: int, retry_seconds: float) -> dict:
    config, path = configured(base, "pilot_seed119", 119)
    run_dir = train_target(state, "pilot_seed119", path, attempts, retry_seconds, True)
    target_summary = run_dir / "target_selection_metrics.json"
    target_accuracy = accuracy(target_summary)
    thresholds = json.loads(PUBLISHED_THRESHOLDS.read_text(encoding="utf-8"))
    minimum = float(thresholds["baselines"][PUBLISHED_BASELINE]["accuracy"])
    accuracy_gate = {
        "status": "PASS" if target_accuracy > minimum else "FAIL",
        "observed_accuracy": target_accuracy,
        "required_strictly_greater_than": minimum,
    }
    save_json(run_dir / "accuracy_gate.json", accuracy_gate)
    if accuracy_gate["status"] != "PASS":
        raise RuntimeError("pilot utility does not beat published Noise_ARL+CEM")

    checkpoint = run_dir / "checkpoint_best.pt"
    decoder = train_attack(
        state,
        "pilot_seed119",
        checkpoint,
        "residual_decoder",
        10_119,
        50,
        attempts,
        retry_seconds,
        group="published_gate_attacks",
    )
    gan = train_attack(
        state,
        "pilot_seed119",
        checkpoint,
        "gan",
        10_119,
        150,
        attempts,
        retry_seconds,
        group="published_gate_attacks",
    )
    output = run_dir / "published_pareto_gate.json"
    run_job(
        state,
        "pilot_published_pareto_gate",
        [
            sys.executable,
            "scripts/check_published_pareto_gate.py",
            "--target-summary",
            str(target_summary),
            "--decoder-metrics",
            str(decoder),
            "--gan-metrics",
            str(gan),
            "--baseline",
            PUBLISHED_BASELINE,
            "--output",
            str(output),
        ],
        output,
        [output],
        1,
        0,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("pilot does not strictly dominate the published baseline")
    return config


def formal_specs(
    base: dict, *, include_matched_baselines: bool = False
) -> list[tuple[str, dict, Path, str]]:
    specs = []
    for seed in range(125, 130):
        config, path = configured(
            base,
            f"headline/proposed_seed{seed}",
            seed,
            {"defense.variant": "identity_token_cem"},
        )
        specs.append((f"headline/proposed_seed{seed}", config, path, "headline"))

    ablations = {
        "no_cem": {
            "defense.name": "gaussian",
            "defense.variant": "identity_token_no_cem",
        },
        "no_margin": {
            "model.margin_head.enabled": False,
            "training.margin_weight": 0.0,
            "defense.variant": "identity_token_no_margin",
        },
        "one_iteration": {
            "model.slot_iterations": 1,
            "defense.variant": "identity_token_one_iteration",
        },
        "single_token": {
            "model.num_slots": 1,
            "defense.regularizer.num_slots": 1,
            "defense.variant": "identity_token_single_token",
        },
    }
    for ablation, overrides in ablations.items():
        for seed in range(125, 128):
            name = f"ablations/{ablation}_seed{seed}"
            config, path = configured(base, name, seed, overrides)
            specs.append((name, config, path, "ablation"))

    if include_matched_baselines:
        for seed in range(125, 128):
            name = f"baseline/official_cem_seed{seed}"
            overrides = {
                "model.name": "vgg11_bn_split",
                "model.pretrained": False,
                "model.bottleneck_channels": 8,
                "model.margin_head.enabled": False,
                "defense.name": "official_cem",
                "defense.variant": "official_cem_reimplementation",
                "training.epochs": 240,
                "training.optimizer": "sgd",
                "training.learning_rate": 0.05,
                "training.weight_decay": 0.0005,
                "training.label_smoothing": 0.0,
                "training.gradient_composition": "sum",
                "training.margin_weight": 0.0,
            }
            config, path = configured(base, name, seed, overrides)
            config["model"]["cut_index"] = 8
            config["training"]["momentum"] = 0.9
            config["defense"]["official"] = {
                "num_clusters": 5,
                "variance_threshold": 0.15,
                "gamma": 0.01,
                "kmeans_iterations": 50,
                "batch_variance_scale": 10.0,
            }
            path = write_config(name, config)
            specs.append((name, config, path, "baseline"))
    return specs


def run_formal(
    state: dict,
    base: dict,
    attempts: int,
    retry_seconds: float,
    *,
    include_matched_baselines: bool = False,
) -> None:
    specs = formal_specs(
        base, include_matched_baselines=include_matched_baselines
    )
    run_records = []
    for name, config, path, tier in specs:
        run_dir = train_target(state, name, path, attempts, retry_seconds, False)
        run_records.append((name, config, run_dir, tier))

    attack_epochs = {
        "conv_decoder": 50,
        "residual_decoder": 50,
        "gan": 150,
        "adaptive": 100,
    }
    for name, _config, run_dir, tier in run_records:
        checkpoint = run_dir / "checkpoint_best.pt"
        if tier in {"headline", "baseline"}:
            attack_types = tuple(attack_epochs)
            seeds = (10_125, 20_125, 30_125)
        else:
            attack_types = ("residual_decoder",)
            seeds = (10_125,)
        for attack_type in attack_types:
            for seed in seeds:
                train_attack(
                    state,
                    name.replace("/", "_"),
                    checkpoint,
                    attack_type,
                    seed,
                    attack_epochs[attack_type],
                    attempts,
                    retry_seconds,
                )

    for name, _config, run_dir, _tier in run_records:
        run_job(
            state,
            f"profile_{name.replace('/', '_')}",
            [sys.executable, "scripts/profile_checkpoint.py", "--checkpoint", str(run_dir / "checkpoint_best.pt")],
            run_dir / "efficiency_profile.json",
            [run_dir / "efficiency_profile.json"],
            attempts,
            retry_seconds,
        )

    summary_prefix = RESULTS_ROOT / "formal_summary"
    run_job(
        state,
        "aggregate_formal_results",
        [sys.executable, "scripts/aggregate_results.py", "--results-root", str(RESULTS_ROOT), "--output-prefix", str(summary_prefix)],
        summary_prefix.with_name(summary_prefix.name + "_target_level").with_suffix(".csv"),
        [
            summary_prefix.with_suffix(".csv"),
            summary_prefix.with_name(summary_prefix.name + "_target_level").with_suffix(".csv"),
            summary_prefix.with_suffix(".json"),
        ],
        attempts,
        retry_seconds,
    )
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    target_csv = summary_prefix.with_name(summary_prefix.name + "_target_level").with_suffix(".csv")
    formal_gate = formal_published_gate(target_csv)
    if include_matched_baselines:
        run_job(
            state,
            "statistical_analysis",
            [
                sys.executable,
                "scripts/statistical_analysis.py",
                "--target-level-csv",
                str(target_csv),
                "--proposed-pattern",
                "identity_token_cem",
                "--baseline-pattern",
                "official_cem_reimplementation",
                "--output",
                str(PAPER_ROOT / "statistical_analysis.json"),
            ],
            PAPER_ROOT / "statistical_analysis.json",
            [PAPER_ROOT / "statistical_analysis.json"],
            attempts,
            retry_seconds,
        )
    run_job(
        state,
        "generate_paper_assets",
        [sys.executable, "scripts/generate_paper_assets.py", "--target-level-csv", str(target_csv), "--output-dir", str(PAPER_ROOT)],
        PAPER_ROOT / "provenance.json",
        [
            PAPER_ROOT / "results_table.tex",
            PAPER_ROOT / "efficiency_table.tex",
            PAPER_ROOT / "privacy_utility.pdf",
            PAPER_ROOT / "provenance.json",
        ],
        attempts,
        retry_seconds,
    )
    manifest = {}
    for root in (RESULTS_ROOT, PAPER_ROOT, LOG_ROOT):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                manifest[str(path.relative_to(ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    save_json(
        RESULTS_ROOT / "publication_materials_manifest.json",
        {"schema_version": 1, "generated_at_utc": now(), "files": manifest},
    )
    if formal_gate["status"] != "PASS":
        raise RuntimeError(
            "formal mean result does not strictly dominate published Bottleneck+CEM"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "facescrub_official")
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal", "all"), default="all")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument(
        "--include-matched-baselines",
        action="store_true",
        help="also train matched Official CEM controls during the formal stage",
    )
    args = parser.parse_args()
    if args.attempts <= 0 or args.retry_seconds < 0:
        raise ValueError("invalid retry policy")

    state = load_state()
    environment = validate_environment(args.data_root)
    state["environment"] = environment
    save_json(STATE_PATH, state)
    base = json.loads(
        (ROOT / "configs" / "facescrub_identity_token_pilot.json").read_text(encoding="utf-8")
    )
    base["data"]["root"] = str(args.data_root)

    if args.stage in {"smoke", "all"}:
        run_smoke(state, base, args.attempts, args.retry_seconds)
        if args.stage == "smoke":
            return
    if args.stage in {"pilot", "all"}:
        run_pilot(state, base, args.attempts, args.retry_seconds)
        if args.stage == "pilot":
            return
    if args.stage in {"formal", "all"}:
        gate = RESULTS_ROOT / "pilot_seed119" / "published_pareto_gate.json"
        if not gate.is_file() or json.loads(gate.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("formal stage is blocked until the published Pareto gate passes")
        run_formal(
            state,
            base,
            args.attempts,
            args.retry_seconds,
            include_matched_baselines=args.include_matched_baselines,
        )
    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()
