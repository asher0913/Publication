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
RESULTS_ROOT = ROOT / "results" / "dual_path_publication"
LOG_ROOT = ROOT / "run_logs" / "dual_path_publication"
STATE_PATH = ROOT / "run_state" / "dual_path_publication.json"
PAPER_OUTPUT = ROOT / "paper" / "generated" / "dual_path"
TARGET_SEEDS = (126, 127, 128, 129, 130)
ATTACKER_SEEDS = (10125, 20125, 30125)
EFFECTIVE_LEGACY_NOISE = 0.31
EFFECTIVE_SEMANTIC_NOISE = 0.10


class ScientificGateFailure(RuntimeError):
    pass


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
        if payload.get("status") == "FAIL":
            return False
    return True


def is_under(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def reset_paths(paths: list[Path]) -> None:
    for path in paths:
        if not any(is_under(path, root) for root in (RESULTS_ROOT, PAPER_OUTPUT)):
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
                status="PASS", finished_at_utc=now(), returncode=0
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
        if result.returncode == 2 and completion.is_file():
            try:
                failed_payload = json.loads(completion.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                failed_payload = {}
            if failed_payload.get("status") == "FAIL":
                state["jobs"][name]["status"] = "SCIENTIFIC_GATE_FAIL"
                save_json(state_path, state)
                raise ScientificGateFailure(
                    f"pre-specified scientific gate failed: {name}"
                )
        if attempt < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"job failed after {attempts} attempts: {name}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_environment(data_root: Path, legacy_checkpoint: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    required_data = (data_root / "train", data_root / "val")
    missing_data = [str(path) for path in required_data if not path.is_dir()]
    required_checkpoint = (
        legacy_checkpoint / "checkpoint_f_best.tar",
        legacy_checkpoint / "checkpoint_cloud_best.tar",
        legacy_checkpoint / "checkpoint_classifier_best.tar",
    )
    missing_checkpoint = [str(path) for path in required_checkpoint if not path.is_file()]
    if missing_data or missing_checkpoint:
        raise FileNotFoundError(
            f"missing data={missing_data}, missing legacy checkpoint={missing_checkpoint}"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    record = {
        "status": "PASS",
        "validated_at_utc": now(),
        "data_root": str(data_root.resolve()),
        "legacy_checkpoint": str(legacy_checkpoint.resolve()),
        "legacy_checkpoint_hashes": {
            path.name: file_sha256(path) for path in required_checkpoint
        },
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "code_revision": revision,
        "target_seeds": list(TARGET_SEEDS),
        "attacker_seeds": list(ATTACKER_SEEDS),
        "effective_legacy_noise_std": EFFECTIVE_LEGACY_NOISE,
        "effective_semantic_noise_std": EFFECTIVE_SEMANTIC_NOISE,
    }
    save_json(RESULTS_ROOT / "environment_gate.json", record)
    return record


def train_target(
    state: dict,
    state_path: Path,
    data_root: Path,
    legacy_checkpoint: Path,
    seed: int,
    attempts: int,
    retry_seconds: float,
) -> Path:
    target_root = RESULTS_ROOT / "targets" / f"seed{seed}"
    stage1 = target_root / "stage1"
    stage2 = target_root / "stage2"
    run_job(
        state,
        state_path,
        f"target_seed{seed}_stage1",
        [
            sys.executable,
            "scripts/train_dual_path_facescrub.py",
            "--legacy-checkpoint-dir",
            str(legacy_checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(stage1),
            "--epochs",
            "80",
            "--batch-size",
            "128",
            "--workers",
            "8",
            "--learning-rate",
            "0.0003",
            "--backbone-learning-rate",
            "0.00005",
            "--legacy-noise-std",
            "0.025",
            "--semantic-noise-std",
            "0.05",
            "--token-dim",
            "256",
            "--semantic-loss-weight",
            "0.35",
            "--utility-threshold",
            "0.0",
            "--seed",
            str(seed),
        ],
        stage1 / "utility_gate.json",
        [stage1],
        attempts,
        retry_seconds,
    )
    run_job(
        state,
        state_path,
        f"target_seed{seed}_stage2",
        [
            sys.executable,
            "scripts/train_dual_path_facescrub.py",
            "--legacy-checkpoint-dir",
            str(legacy_checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(stage2),
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
            "0.22",
            "--semantic-noise-std",
            "0.10",
            "--token-dim",
            "256",
            "--semantic-loss-weight",
            "0.50",
            "--initial-dual-path-checkpoint",
            str(stage1 / "checkpoint_best.pt"),
            "--train-legacy-server",
            "--legacy-server-learning-rate",
            "0.0001",
            "--seed",
            str(seed),
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
    checkpoint: Path,
    stage1_checkpoint: Path,
    data_root: Path,
    target_seed: int,
    attempts: int,
    retry_seconds: float,
) -> None:
    target_root = RESULTS_ROOT / "targets" / f"seed{target_seed}"
    utility_output = target_root / "utility_l031_s010.json"
    evaluation_seeds = [100_000 + target_seed + offset for offset in range(3)]
    stage1_output = target_root / "utility_stage1_l031_s010.json"
    run_job(
        state,
        state_path,
        f"utility_stage1_seed{target_seed}",
        [
            sys.executable,
            "scripts/evaluate_dual_path_utility.py",
            "--dual-path-checkpoint",
            str(stage1_checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(stage1_output),
            "--legacy-noise-std",
            str(EFFECTIVE_LEGACY_NOISE),
            "--semantic-noise-std",
            str(EFFECTIVE_SEMANTIC_NOISE),
            "--evaluation-seeds",
            *[str(seed) for seed in evaluation_seeds],
            "--batch-size",
            "256",
            "--workers",
            "8",
            "--utility-threshold",
            "0.0",
        ],
        stage1_output,
        [stage1_output],
        attempts,
        retry_seconds,
    )
    run_job(
        state,
        state_path,
        f"utility_seed{target_seed}",
        [
            sys.executable,
            "scripts/evaluate_dual_path_utility.py",
            "--dual-path-checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(utility_output),
            "--legacy-noise-std",
            str(EFFECTIVE_LEGACY_NOISE),
            "--semantic-noise-std",
            str(EFFECTIVE_SEMANTIC_NOISE),
            "--evaluation-seeds",
            *[str(seed) for seed in evaluation_seeds],
            "--batch-size",
            "256",
            "--workers",
            "8",
        ],
        utility_output,
        [utility_output],
        attempts,
        retry_seconds,
    )
    profile_output = target_root / "efficiency_profile.json"
    run_job(
        state,
        state_path,
        f"profile_seed{target_seed}",
        [
            sys.executable,
            "scripts/profile_dual_path_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output",
            str(profile_output),
        ],
        profile_output,
        [profile_output],
        attempts,
        retry_seconds,
    )


def train_attack(
    state: dict,
    state_path: Path,
    checkpoint: Path,
    data_root: Path,
    target_seed: int,
    attacker_seed: int,
    attack: str,
    knowledge: str,
    attempts: int,
    retry_seconds: float,
) -> Path:
    output_dir = (
        RESULTS_ROOT
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
        str(output_dir),
        "--attack-type",
        "gan" if attack == "gan" else "decoder",
        "--attack-knowledge",
        knowledge,
        "--epochs",
        "100" if attack == "adaptive" else "50",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--width",
        "192" if attack == "adaptive" else "128",
        "--residual-blocks",
        "8" if attack == "adaptive" else "4",
        "--legacy-noise-std",
        str(EFFECTIVE_LEGACY_NOISE),
        "--semantic-noise-std",
        str(EFFECTIVE_SEMANTIC_NOISE),
        "--seed",
        str(attacker_seed),
        "--evaluation-noise-seeds",
        str(500_000 + target_seed),
        str(600_000 + target_seed),
        str(700_000 + target_seed),
        "--perceptual-metrics",
    ]
    if attack == "gan":
        decoder = output_dir.parent / f"decoder_{knowledge}" / "checkpoint_best.pt"
        command.extend(("--initial-decoder-checkpoint", str(decoder)))
    run_job(
        state,
        state_path,
        f"attack_t{target_seed}_a{attacker_seed}_{attack}_{knowledge}",
        command,
        output_dir / "attack_metrics.json",
        [output_dir],
        attempts,
        retry_seconds,
    )
    return output_dir / "attack_metrics.json"


def run_attacks(
    state: dict,
    state_path: Path,
    checkpoints: dict[int, Path],
    data_root: Path,
    attempts: int,
    retry_seconds: float,
) -> None:
    for target_seed, checkpoint in checkpoints.items():
        for attacker_seed in ATTACKER_SEEDS:
            for knowledge in ("training", "inference"):
                train_attack(
                    state,
                    state_path,
                    checkpoint,
                    data_root,
                    target_seed,
                    attacker_seed,
                    "decoder",
                    knowledge,
                    attempts,
                    retry_seconds,
                )
                train_attack(
                    state,
                    state_path,
                    checkpoint,
                    data_root,
                    target_seed,
                    attacker_seed,
                    "gan",
                    knowledge,
                    attempts,
                    retry_seconds,
                )
        for knowledge in ("training", "inference"):
            train_attack(
                state,
                state_path,
                checkpoint,
                data_root,
                target_seed,
                ATTACKER_SEEDS[0],
                "adaptive",
                knowledge,
                attempts,
                retry_seconds,
            )


def aggregate(
    state: dict,
    state_path: Path,
    attempts: int,
    retry_seconds: float,
) -> None:
    output = RESULTS_ROOT / "formal_summary.json"
    run_job(
        state,
        state_path,
        "aggregate_and_gate",
        [
            sys.executable,
            "scripts/aggregate_dual_path_publication.py",
            "--results-root",
            str(RESULTS_ROOT),
            "--paper-output",
            str(PAPER_OUTPUT),
            "--target-seeds",
            *[str(seed) for seed in TARGET_SEEDS],
            "--attacker-seeds",
            *[str(seed) for seed in ATTACKER_SEEDS],
        ],
        output,
        [
            output,
            RESULTS_ROOT / "utility_target_level.csv",
            RESULTS_ROOT / "attack_runs.csv",
            RESULTS_ROOT / "publication_materials_manifest.json",
            PAPER_OUTPUT / "results_table.tex",
            PAPER_OUTPUT / "attacker_table.tex",
            PAPER_OUTPUT / "efficiency_table.tex",
            PAPER_OUTPUT / "ablation_table.tex",
            PAPER_OUTPUT / "privacy_utility.pdf",
            PAPER_OUTPUT / "privacy_utility.png",
            PAPER_OUTPUT / "main_results_narrative.tex",
            PAPER_OUTPUT / "attack_results_narrative.tex",
            PAPER_OUTPUT / "ablation_results_narrative.tex",
            PAPER_OUTPUT / "efficiency_results_narrative.tex",
            PAPER_OUTPUT / "abstract_results.tex",
            PAPER_OUTPUT / "conclusion_results.tex",
        ],
        1,
        retry_seconds,
    )


def generate_qualitative(
    state: dict,
    state_path: Path,
    checkpoint: Path,
    data_root: Path,
    target_seed: int,
    attempts: int,
    retry_seconds: float,
) -> None:
    attacker_root = (
        RESULTS_ROOT
        / "attacks"
        / f"target_seed{target_seed}"
        / "attacker_seed10125"
    )
    output = PAPER_OUTPUT / "qualitative_grid.png"
    run_job(
        state,
        state_path,
        "qualitative_grid",
        [
            sys.executable,
            "scripts/export_dual_path_qualitative_grid.py",
            "--target-checkpoint",
            str(checkpoint),
            "--decoder-checkpoint",
            str(attacker_root / "decoder_inference" / "checkpoint_best.pt"),
            "--gan-checkpoint",
            str(attacker_root / "gan_inference" / "checkpoint_best.pt"),
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--samples",
            "8",
            "--noise-seed",
            str(500_000 + target_seed),
        ],
        output,
        [output, output.with_suffix(".provenance.json")],
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
    environment = validate_environment(args.data_root, args.legacy_checkpoint_dir)
    state["environment"] = environment
    save_json(state_path, state)

    checkpoints: dict[int, Path] = {}
    stage1_checkpoints: dict[int, Path] = {}
    for seed in TARGET_SEEDS:
        expected = RESULTS_ROOT / "targets" / f"seed{seed}" / "stage2" / "checkpoint_best.pt"
        if args.stage in ("targets", "all"):
            checkpoints[seed] = train_target(
                state,
                state_path,
                args.data_root,
                args.legacy_checkpoint_dir,
                seed,
                args.attempts,
                args.retry_seconds,
            )
            stage1_checkpoints[seed] = (
                RESULTS_ROOT / "targets" / f"seed{seed}" / "stage1" / "checkpoint_best.pt"
            )
        else:
            if not expected.is_file():
                raise FileNotFoundError(f"target stage is incomplete: {expected}")
            checkpoints[seed] = expected
            stage1_expected = (
                RESULTS_ROOT / "targets" / f"seed{seed}" / "stage1" / "checkpoint_best.pt"
            )
            if not stage1_expected.is_file():
                raise FileNotFoundError(
                    f"target stage-1 checkpoint is incomplete: {stage1_expected}"
                )
            stage1_checkpoints[seed] = stage1_expected

    if args.stage in ("targets", "all"):
        for seed, checkpoint in checkpoints.items():
            evaluate_target(
                state,
                state_path,
                checkpoint,
                stage1_checkpoints[seed],
                args.data_root,
                seed,
                args.attempts,
                args.retry_seconds,
            )
        if args.stage == "targets":
            return
    if args.stage in ("attacks", "all"):
        run_attacks(
            state,
            state_path,
            checkpoints,
            args.data_root,
            args.attempts,
            args.retry_seconds,
        )
        generate_qualitative(
            state,
            state_path,
            checkpoints[TARGET_SEEDS[0]],
            args.data_root,
            TARGET_SEEDS[0],
            args.attempts,
            args.retry_seconds,
        )
        if args.stage == "attacks":
            return
    if args.stage in ("aggregate", "all"):
        aggregate(state, state_path, args.attempts, args.retry_seconds)

    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    save_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "PUBLICATION_EVIDENCE_READY",
                "summary": str(RESULTS_ROOT / "formal_summary.json"),
                "paper_assets": str(PAPER_OUTPUT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ScientificGateFailure as error:
        print(json.dumps({"status": "SCIENTIFIC_GATE_FAIL", "error": str(error)}))
        raise SystemExit(2) from error
