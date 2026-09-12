#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
DEFAULT_RESULTS = ROOT / "results" / "supervisor_revision"
DEFAULT_TARGETS = ROOT / "results" / "dual_path_publication" / "targets"
DEFAULT_STATE = ROOT / "run_state" / "supervisor_revision.json"
DEFAULT_LOGS = ROOT / "run_logs" / "supervisor_revision"
DEFAULT_TARGET_SEEDS = (126, 127, 128, 129, 130)
DEFAULT_TOKEN_SEEDS = (126, 127, 128)
DEFAULT_TOKEN_DIMENSIONS = (64, 128, 384)
DEFAULT_ATTACKER_SEEDS = (10125, 20125, 30125)
DEFAULT_NOISE_POINTS = ((0.22, 0.10), (0.31, 0.10), (0.38, 0.10))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def completion_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix != ".json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("status") not in {None, "FAIL"}


def run_job(
    state: dict,
    state_path: Path,
    name: str,
    command: list[str],
    completion: Path,
    reset: list[Path],
    log_root: Path,
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
    log_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        for path in reset:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        log_path = log_root / f"{name}.attempt{attempt}.log"
        state["jobs"][name] = {
            "status": "RUNNING",
            "attempt": attempt,
            "command": command,
            "completion": str(completion),
            "log": str(log_path),
            "started_at_utc": now(),
        }
        save_json(state_path, state)
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


def target_checkpoint(targets_root: Path, seed: int) -> Path:
    return targets_root / f"seed{seed}" / "stage2" / "checkpoint_best.pt"


def point_name(legacy_noise: float, semantic_noise: float) -> str:
    return f"g{legacy_noise:.2f}_s{semantic_noise:.2f}".replace(".", "")


def run_fast_evaluations(args, state, state_path, checkpoints: list[Path]) -> None:
    fusion_root = args.results_root / "fusion"
    run_job(
        state,
        state_path,
        "fusion_ablation",
        [
            sys.executable,
            "scripts/evaluate_dual_path_fusion_ablation.py",
            "--dual-path-checkpoints",
            *map(str, checkpoints),
            "--legacy-checkpoint-dir",
            str(args.legacy_checkpoint_dir),
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(fusion_root),
            "--legacy-noise-std",
            "0.31",
            "--semantic-noise-std",
            "0.10",
            "--evaluation-seeds",
            "100126",
            "200126",
            "300126",
        ],
        fusion_root / "fusion_ablation_summary.json",
        [fusion_root],
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )
    noise_root = args.results_root / "noise_utility"
    run_job(
        state,
        state_path,
        "noise_utility_sensitivity",
        [
            sys.executable,
            "scripts/evaluate_dual_path_noise_sensitivity.py",
            "--dual-path-checkpoints",
            *map(str, checkpoints),
            "--legacy-checkpoint-dir",
            str(args.legacy_checkpoint_dir),
            "--data-root",
            str(args.data_root),
            "--legacy-checkpoint-dir",
            str(args.legacy_checkpoint_dir),
            "--output-dir",
            str(noise_root),
            "--legacy-noise",
            *[str(item[0]) for item in args.noise_points],
            "--semantic-noise",
            *[str(item[1]) for item in args.noise_points],
            "--evaluation-seeds",
            "100126",
            "200126",
            "300126",
        ],
        noise_root / "noise_utility_summary.json",
        [noise_root],
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )


def train_token_dimension(args, state, state_path, dimension: int, seed: int) -> Path:
    root = args.results_root / "token_dimension" / f"dim{dimension}" / f"seed{seed}"
    stage1 = root / "stage1"
    stage2 = root / "stage2"
    common = [
        "--legacy-checkpoint-dir",
        str(args.legacy_checkpoint_dir),
        "--data-root",
        str(args.data_root),
        "--batch-size",
        "128",
        "--workers",
        "8",
        "--token-dim",
        str(dimension),
        "--semantic-backbone",
        "mobilenet_v3_large",
        "--utility-threshold",
        "0.0",
        "--seed",
        str(seed),
    ]
    run_job(
        state,
        state_path,
        f"token_dim{dimension}_seed{seed}_stage1",
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
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )
    run_job(
        state,
        state_path,
        f"token_dim{dimension}_seed{seed}_stage2",
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
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )
    utility = root / "utility.json"
    run_job(
        state,
        state_path,
        f"token_dim{dimension}_seed{seed}_utility",
        [
            sys.executable,
            "scripts/evaluate_dual_path_utility.py",
            "--dual-path-checkpoint",
            str(stage2 / "checkpoint_best.pt"),
            "--data-root",
            str(args.data_root),
            "--output",
            str(utility),
            "--legacy-noise-std",
            "0.31",
            "--semantic-noise-std",
            "0.10",
            "--evaluation-seeds",
            str(100000 + seed),
            str(200000 + seed),
            str(300000 + seed),
            "--utility-threshold",
            "0.0",
        ],
        utility,
        [utility],
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )
    return stage2 / "checkpoint_best.pt"


def run_identity_attacks(args, state, state_path) -> None:
    output = args.results_root / "identity_conditioned_attack"
    completion = output / "identity_conditioned_attack_summary.json"
    run_job(
        state,
        state_path,
        "identity_conditioned_attack_protocol",
        [
            sys.executable,
            "scripts/run_identity_conditioned_attack_protocol.py",
            "--data-root",
            str(args.data_root),
            "--targets-root",
            str(args.targets_root),
            "--results-root",
            str(output),
            "--state",
            str(args.results_root / "identity_conditioned_state.json"),
            "--log-root",
            str(args.log_root / "identity_conditioned"),
            "--target-seeds",
            *map(str, args.noise_target_seeds),
            "--attacker-seeds",
            *map(str, args.attacker_seeds),
            "--knowledge",
            "training",
            "inference",
            "--epochs",
            "50",
            "--attempts",
            str(args.attempts),
            "--retry-seconds",
            str(args.retry_seconds),
        ],
        completion,
        [output, args.results_root / "identity_conditioned_state.json"],
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )


def run_noise_attacks(args, state, state_path) -> None:
    for legacy_noise, semantic_noise in args.noise_points:
        name = point_name(legacy_noise, semantic_noise)
        for target_seed in args.noise_target_seeds:
            checkpoint = target_checkpoint(args.targets_root, target_seed)
            for attacker_seed in args.noise_attacker_seeds:
                for knowledge in ("training", "inference"):
                    output = (
                        args.results_root
                        / "noise_attacks"
                        / name
                        / f"target_seed{target_seed}"
                        / f"attacker_seed{attacker_seed}"
                        / f"decoder_{knowledge}"
                    )
                    run_job(
                        state,
                        state_path,
                        f"noise_{name}_t{target_seed}_a{attacker_seed}_{knowledge}",
                        [
                            sys.executable,
                            "scripts/attack_dual_path_facescrub.py",
                            "--dual-path-checkpoint",
                            str(checkpoint),
                            "--legacy-checkpoint-dir",
                            str(args.legacy_checkpoint_dir),
                            "--data-root",
                            str(args.data_root),
                            "--output-dir",
                            str(output),
                            "--attack-type",
                            "decoder",
                            "--attack-view",
                            "joint",
                            "--identity-conditioning",
                            "none",
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
                            str(legacy_noise),
                            "--semantic-noise-std",
                            str(semantic_noise),
                            "--seed",
                            str(attacker_seed),
                            "--evaluation-noise-seeds",
                            str(500000 + target_seed),
                            str(600000 + target_seed),
                            str(700000 + target_seed),
                            "--perceptual-metrics",
                        ],
                        output / "attack_metrics.json",
                        [output],
                        args.log_root,
                        args.attempts,
                        args.retry_seconds,
                    )


def aggregate_all(args, state, state_path) -> None:
    completion = args.results_root / "supervisor_revision_summary.json"
    run_job(
        state,
        state_path,
        "aggregate_supervisor_revision",
        [
            sys.executable,
            "scripts/aggregate_supervisor_revision_experiments.py",
            "--results-root",
            str(args.results_root),
            "--token-dimensions",
            *map(str, args.token_dimensions),
            "--token-seeds",
            *map(str, args.token_seeds),
            "--noise-points",
            *[f"{g},{s}" for g, s in args.noise_points],
            "--noise-target-seeds",
            *map(str, args.noise_target_seeds),
            "--noise-attacker-seeds",
            *map(str, args.noise_attacker_seeds),
            "--knowledge",
            "training",
            "inference",
        ],
        completion,
        [
            completion,
            args.results_root / "token_dimension_table.tex",
            args.results_root / "noise_attack_table.tex",
        ],
        args.log_root,
        args.attempts,
        args.retry_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--legacy-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument(
        "--stage",
        choices=("fast", "targets", "attacks", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--target-seeds", nargs="+", type=int, default=list(DEFAULT_TARGET_SEEDS))
    parser.add_argument("--token-seeds", nargs="+", type=int, default=list(DEFAULT_TOKEN_SEEDS))
    parser.add_argument(
        "--token-dimensions", nargs="+", type=int, default=list(DEFAULT_TOKEN_DIMENSIONS)
    )
    parser.add_argument(
        "--noise-target-seeds", nargs="+", type=int, default=list(DEFAULT_TOKEN_SEEDS)
    )
    parser.add_argument(
        "--attacker-seeds", nargs="+", type=int, default=list(DEFAULT_ATTACKER_SEEDS)
    )
    parser.add_argument("--noise-attacker-seeds", nargs="+", type=int, default=[10125])
    parser.add_argument(
        "--noise-points", nargs="+", default=[f"{g},{s}" for g, s in DEFAULT_NOISE_POINTS]
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parsed_points = []
    for value in args.noise_points:
        legacy, semantic = value.split(",", maxsplit=1)
        parsed_points.append((float(legacy), float(semantic)))
    args.noise_points = parsed_points
    for path in (args.data_root / "train", args.data_root / "val"):
        if not path.is_dir():
            raise FileNotFoundError(path)
    checkpoints = [target_checkpoint(args.targets_root, seed) for seed in args.target_seeds]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing formal target checkpoints: {missing}")
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    state_path = args.state.resolve()
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "started_at_utc": now(), "jobs": {}}
    )
    state["protocol"] = {
        "target_seeds": args.target_seeds,
        "token_dimensions": args.token_dimensions,
        "token_seeds": args.token_seeds,
        "noise_points": args.noise_points,
        "noise_target_seeds": args.noise_target_seeds,
        "identity_attacker_seeds": args.attacker_seeds,
        "noise_attacker_seeds": args.noise_attacker_seeds,
        "cross_dataset_evidence_is_separate": True,
    }
    save_json(state_path, state)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "protocol": state["protocol"]}))
        return

    if args.stage in ("fast", "all"):
        run_fast_evaluations(args, state, state_path, checkpoints)
    if args.stage in ("targets", "all"):
        for dimension in args.token_dimensions:
            for seed in args.token_seeds:
                train_token_dimension(args, state, state_path, dimension, seed)
    if args.stage in ("attacks", "all"):
        run_identity_attacks(args, state, state_path)
        run_noise_attacks(args, state, state_path)
    if args.stage in ("aggregate", "all"):
        aggregate_all(args, state, state_path)
    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    save_json(state_path, state)
    print(json.dumps({"status": "SUPERVISOR_REVISION_EVIDENCE_READY"}))


if __name__ == "__main__":
    main()
