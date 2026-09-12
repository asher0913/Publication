#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "dual_path_supplementary"
MAIN_RESULTS_ROOT = ROOT / "results" / "dual_path_publication"
PAPER_OUTPUT = ROOT / "paper" / "generated" / "dual_path_supplementary"
LOG_ROOT = ROOT / "run_logs" / "dual_path_supplementary"
STATE_PATH = ROOT / "run_state" / "dual_path_supplementary.json"
CAPACITY_SEEDS = (126, 127, 128, 129, 130)
BACKBONE_SEEDS = (126, 127, 128)
ATTACKER_SEEDS = (10125, 20125, 30125)
LEGACY_NOISE = 0.31
SEMANTIC_NOISE = 0.10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "started_at_utc": now(), "jobs": {}}


def completion_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return payload.get("status") != "FAIL"
    return True


def under_campaign(path: Path) -> bool:
    resolved = path.resolve()
    return any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in (RESULTS_ROOT, PAPER_OUTPUT)
    )


def reset_paths(paths: list[Path]) -> None:
    for path in paths:
        if not under_campaign(path):
            raise ValueError(f"refusing to reset path outside campaign roots: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def run_job(
    state: dict,
    state_path: Path,
    name: str,
    command: list[str],
    completion: Path,
    reset: list[Path],
    attempts: int,
    retry_seconds: float,
) -> None:
    if completion_valid(completion):
        state["jobs"][name] = {
            "status": "PASS",
            "completion": str(completion),
            "recovered_from_artifact": True,
        }
        save_json(state_path, state)
        print(json.dumps({"job": name, "status": "SKIP_COMPLETE"}), flush=True)
        return
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        reset_paths(reset)
        log_path = LOG_ROOT / f"{name}.attempt{attempt}.log"
        state["jobs"][name] = {
            "status": "RUNNING",
            "attempt": attempt,
            "command": command,
            "completion": str(completion),
            "log": str(log_path),
            "started_at_utc": now(),
        }
        save_json(state_path, state)
        print(
            json.dumps({"job": name, "status": "RUNNING", "attempt": attempt}),
            flush=True,
        )
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
            state["jobs"][name].update(
                status="PASS", returncode=0, finished_at_utc=now()
            )
            save_json(state_path, state)
            print(json.dumps({"job": name, "status": "PASS"}), flush=True)
            return
        state["jobs"][name].update(
            status="RETRY" if attempt < attempts else "FAIL",
            returncode=result.returncode,
            finished_at_utc=now(),
        )
        save_json(state_path, state)
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"job failed after {attempts} attempts: {name}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_environment(data_root: Path, legacy_checkpoint: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    required = [
        data_root / "train",
        data_root / "val",
        legacy_checkpoint / "checkpoint_f_best.tar",
        legacy_checkpoint / "checkpoint_cloud_best.tar",
        legacy_checkpoint / "checkpoint_classifier_best.tar",
        MAIN_RESULTS_ROOT / "formal_summary.json",
    ]
    required.extend(
        MAIN_RESULTS_ROOT
        / "targets"
        / f"seed{seed}"
        / "stage1"
        / "checkpoint_best.pt"
        for seed in CAPACITY_SEEDS
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing supplementary prerequisite(s): {missing}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "status": "PASS",
        "validated_at_utc": now(),
        "code_revision": revision,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "data_root": str(data_root.resolve()),
        "legacy_checkpoint": str(legacy_checkpoint.resolve()),
        "legacy_checkpoint_sha256": file_sha256(
            legacy_checkpoint / "checkpoint_f_best.tar"
        ),
        "capacity_matched_target_seeds": list(CAPACITY_SEEDS),
        "cross_backbone_target_seeds": list(BACKBONE_SEEDS),
        "attacker_seeds": list(ATTACKER_SEEDS),
        "effective_legacy_noise_std": LEGACY_NOISE,
        "effective_semantic_noise_std": SEMANTIC_NOISE,
    }
    save_json(RESULTS_ROOT / "environment_gate.json", payload)
    return payload


def train_capacity_control(
    state: dict,
    state_path: Path,
    data_root: Path,
    legacy_checkpoint: Path,
    seed: int,
    attempts: int,
    retry_seconds: float,
) -> Path:
    output = RESULTS_ROOT / "capacity_matched" / "targets" / f"seed{seed}" / "target"
    initial = (
        MAIN_RESULTS_ROOT
        / "targets"
        / f"seed{seed}"
        / "stage1"
        / "checkpoint_best.pt"
    )
    run_job(
        state,
        state_path,
        f"capacity_target_seed{seed}",
        [
            sys.executable,
            "scripts/train_dual_path_facescrub.py",
            "--legacy-checkpoint-dir",
            str(legacy_checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output),
            "--epochs",
            "40",
            "--batch-size",
            "128",
            "--workers",
            "8",
            "--learning-rate",
            "0.00015",
            "--backbone-learning-rate",
            "0.00002",
            "--legacy-noise-std",
            "0.025",
            "--semantic-noise-std",
            "0.05",
            "--token-dim",
            "256",
            "--semantic-backbone",
            "mobilenet_v3_large",
            "--semantic-loss-weight",
            "0.35",
            "--initial-dual-path-checkpoint",
            str(initial),
            "--utility-threshold",
            "0.0",
            "--seed",
            str(seed),
        ],
        output / "utility_gate.json",
        [output],
        attempts,
        retry_seconds,
    )
    return output / "checkpoint_best.pt"


def train_resnet_control(
    state: dict,
    state_path: Path,
    data_root: Path,
    legacy_checkpoint: Path,
    seed: int,
    attempts: int,
    retry_seconds: float,
) -> Path:
    target_root = RESULTS_ROOT / "resnet18" / "targets" / f"seed{seed}"
    stage1 = target_root / "stage1"
    stage2 = target_root / "stage2"
    common = [
        "--legacy-checkpoint-dir",
        str(legacy_checkpoint),
        "--data-root",
        str(data_root),
        "--batch-size",
        "128",
        "--workers",
        "8",
        "--token-dim",
        "256",
        "--semantic-backbone",
        "resnet18",
        "--utility-threshold",
        "0.0",
        "--seed",
        str(seed),
    ]
    run_job(
        state,
        state_path,
        f"resnet18_target_seed{seed}_stage1",
        [
            sys.executable,
            "scripts/train_dual_path_facescrub.py",
            *common,
            "--output-dir",
            str(stage1),
            "--epochs",
            "80",
            "--learning-rate",
            "0.0003",
            "--backbone-learning-rate",
            "0.00005",
            "--legacy-noise-std",
            "0.025",
            "--semantic-noise-std",
            "0.05",
            "--semantic-loss-weight",
            "0.35",
        ],
        stage1 / "utility_gate.json",
        [stage1],
        attempts,
        retry_seconds,
    )
    run_job(
        state,
        state_path,
        f"resnet18_target_seed{seed}_stage2",
        [
            sys.executable,
            "scripts/train_dual_path_facescrub.py",
            *common,
            "--output-dir",
            str(stage2),
            "--epochs",
            "40",
            "--learning-rate",
            "0.00015",
            "--backbone-learning-rate",
            "0.00002",
            "--legacy-noise-std",
            "0.22",
            "--semantic-noise-std",
            "0.10",
            "--semantic-loss-weight",
            "0.50",
            "--initial-dual-path-checkpoint",
            str(stage1 / "checkpoint_best.pt"),
            "--train-legacy-server",
            "--legacy-server-learning-rate",
            "0.0001",
        ],
        stage2 / "utility_gate.json",
        [stage2],
        attempts,
        retry_seconds,
    )
    return stage2 / "checkpoint_best.pt"


def evaluate_target(
    state: dict,
    state_path: Path,
    method: str,
    checkpoint: Path,
    data_root: Path,
    seed: int,
    attempts: int,
    retry_seconds: float,
) -> None:
    target_root = RESULTS_ROOT / method / "targets" / f"seed{seed}"
    utility = target_root / "utility_l031_s010.json"
    profile = target_root / "efficiency_profile.json"
    evaluation_seeds = [100_000 + seed + offset for offset in range(3)]
    run_job(
        state,
        state_path,
        f"{method}_utility_seed{seed}",
        [
            sys.executable,
            "scripts/evaluate_dual_path_utility.py",
            "--dual-path-checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(utility),
            "--legacy-noise-std",
            str(LEGACY_NOISE),
            "--semantic-noise-std",
            str(SEMANTIC_NOISE),
            "--evaluation-seeds",
            *[str(value) for value in evaluation_seeds],
            "--batch-size",
            "256",
            "--workers",
            "8",
            "--utility-threshold",
            "0.0",
        ],
        utility,
        [utility],
        attempts,
        retry_seconds,
    )
    run_job(
        state,
        state_path,
        f"{method}_profile_seed{seed}",
        [
            sys.executable,
            "scripts/profile_dual_path_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(profile),
        ],
        profile,
        [profile],
        attempts,
        retry_seconds,
    )


def run_attack(
    state: dict,
    state_path: Path,
    method: str,
    checkpoint: Path,
    data_root: Path,
    target_seed: int,
    attacker_seed: int,
    attack: str,
    knowledge: str,
    attempts: int,
    retry_seconds: float,
) -> None:
    output = (
        RESULTS_ROOT
        / method
        / "attacks"
        / f"target_seed{target_seed}"
        / f"attacker_seed{attacker_seed}"
        / f"{attack}_{knowledge}"
    )
    command = [
        sys.executable,
        "scripts/attack_dual_path_facescrub.py",
        "--dual-path-checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output),
        "--attack-type",
        attack,
        "--attack-knowledge",
        knowledge,
        "--epochs",
        "50",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--width",
        "128",
        "--residual-blocks",
        "4",
        "--legacy-noise-std",
        str(LEGACY_NOISE),
        "--semantic-noise-std",
        str(SEMANTIC_NOISE),
        "--seed",
        str(attacker_seed),
        "--evaluation-noise-seeds",
        str(500_000 + target_seed),
        str(600_000 + target_seed),
        str(700_000 + target_seed),
        "--perceptual-metrics",
    ]
    if attack == "gan":
        decoder = output.parent / f"decoder_{knowledge}" / "checkpoint_best.pt"
        command.extend(("--initial-decoder-checkpoint", str(decoder)))
    run_job(
        state,
        state_path,
        f"{method}_attack_t{target_seed}_a{attacker_seed}_{attack}_{knowledge}",
        command,
        output / "attack_metrics.json",
        [output],
        attempts,
        retry_seconds,
    )


def aggregate_and_build(
    state: dict,
    state_path: Path,
    attempts: int,
    retry_seconds: float,
) -> None:
    summary = RESULTS_ROOT / "supplementary_summary.json"
    run_job(
        state,
        state_path,
        "aggregate_supplementary",
        [
            sys.executable,
            "scripts/aggregate_dual_path_supplementary.py",
            "--results-root",
            str(RESULTS_ROOT),
            "--main-results-root",
            str(MAIN_RESULTS_ROOT),
            "--paper-output",
            str(PAPER_OUTPUT),
            "--matched-target-seeds",
            *[str(seed) for seed in CAPACITY_SEEDS],
            "--backbone-target-seeds",
            *[str(seed) for seed in BACKBONE_SEEDS],
            "--attacker-seeds",
            *[str(seed) for seed in ATTACKER_SEEDS],
        ],
        summary,
        [
            summary,
            RESULTS_ROOT / "utility_target_level.csv",
            RESULTS_ROOT / "attack_runs.csv",
            RESULTS_ROOT / "supplementary_materials_manifest.json",
            PAPER_OUTPUT,
        ],
        attempts,
        retry_seconds,
    )
    completion = RESULTS_ROOT / "paper_build.json"
    run_job(
        state,
        state_path,
        "build_pattern_recognition_paper",
        [
            sys.executable,
            "scripts/finalize_dual_path_supplementary.py",
            "--summary",
            str(summary),
            "--output",
            str(completion),
        ],
        completion,
        [completion],
        attempts,
        retry_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--legacy-checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("targets", "attacks", "aggregate", "all"), default="all"
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    args = parser.parse_args()
    if args.attempts <= 0 or args.retry_seconds < 0:
        raise ValueError("invalid retry policy")

    state_path = args.state.resolve()
    state = load_state(state_path)
    state["environment"] = validate_environment(
        args.data_root, args.legacy_checkpoint_dir
    )
    save_json(state_path, state)
    checkpoints: dict[str, dict[int, Path]] = {
        "capacity_matched": {},
        "resnet18": {},
    }
    for seed in CAPACITY_SEEDS:
        expected = (
            RESULTS_ROOT
            / "capacity_matched"
            / "targets"
            / f"seed{seed}"
            / "target"
            / "checkpoint_best.pt"
        )
        checkpoints["capacity_matched"][seed] = (
            train_capacity_control(
                state,
                state_path,
                args.data_root,
                args.legacy_checkpoint_dir,
                seed,
                args.attempts,
                args.retry_seconds,
            )
            if args.stage in ("targets", "all")
            else expected
        )
    for seed in BACKBONE_SEEDS:
        expected = (
            RESULTS_ROOT
            / "resnet18"
            / "targets"
            / f"seed{seed}"
            / "stage2"
            / "checkpoint_best.pt"
        )
        checkpoints["resnet18"][seed] = (
            train_resnet_control(
                state,
                state_path,
                args.data_root,
                args.legacy_checkpoint_dir,
                seed,
                args.attempts,
                args.retry_seconds,
            )
            if args.stage in ("targets", "all")
            else expected
        )
    for method, method_checkpoints in checkpoints.items():
        for seed, checkpoint in method_checkpoints.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if args.stage in ("targets", "all"):
                evaluate_target(
                    state,
                    state_path,
                    method,
                    checkpoint,
                    args.data_root,
                    seed,
                    args.attempts,
                    args.retry_seconds,
                )
    if args.stage == "targets":
        return

    if args.stage in ("attacks", "all"):
        for method, method_checkpoints in checkpoints.items():
            for target_seed, checkpoint in method_checkpoints.items():
                for attacker_seed in ATTACKER_SEEDS:
                    for knowledge in ("training", "inference"):
                        run_attack(
                            state,
                            state_path,
                            method,
                            checkpoint,
                            args.data_root,
                            target_seed,
                            attacker_seed,
                            "decoder",
                            knowledge,
                            args.attempts,
                            args.retry_seconds,
                        )
                        run_attack(
                            state,
                            state_path,
                            method,
                            checkpoint,
                            args.data_root,
                            target_seed,
                            attacker_seed,
                            "gan",
                            knowledge,
                            args.attempts,
                            args.retry_seconds,
                        )
    if args.stage == "attacks":
        return
    if args.stage in ("aggregate", "all"):
        aggregate_and_build(state, state_path, args.attempts, args.retry_seconds)
    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    save_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "SUPPLEMENTARY_EVIDENCE_READY",
                "summary": str(RESULTS_ROOT / "supplementary_summary.json"),
                "paper": str(MAIN_RESULTS_ROOT / "Pattern_Recognition_DualPath_CEM.pdf"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
