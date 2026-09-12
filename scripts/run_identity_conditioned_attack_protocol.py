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
DEFAULT_RESULTS = ROOT / "results" / "identity_conditioned_attack"
DEFAULT_TARGETS = ROOT / "results" / "dual_path_publication" / "targets"
DEFAULT_STATE = ROOT / "run_state" / "identity_conditioned_attack.json"
DEFAULT_LOGS = ROOT / "run_logs" / "identity_conditioned_attack"
SCENARIOS = {
    "g_only": ("spatial", "uniform"),
    "joint_unconditioned": ("joint", "uniform"),
    "joint_predicted_identity": ("joint", "predicted"),
    "joint_oracle_identity": ("joint", "oracle"),
}


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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return "evaluation" in payload and payload.get("identity_conditioning") is not None


def run_job(
    state: dict,
    state_path: Path,
    name: str,
    command: list[str],
    output_dir: Path,
    log_root: Path,
    attempts: int,
    retry_seconds: float,
) -> None:
    completion = output_dir / "attack_metrics.json"
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
        if output_dir.exists():
            shutil.rmtree(output_dir)
        log_path = log_root / f"{name}.attempt{attempt}.log"
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
            state["jobs"][name].update(status="PASS", returncode=0, finished_at_utc=now())
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        help="override the machine-specific legacy path stored in target checkpoints",
    )
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOGS)
    parser.add_argument(
        "--target-seeds", nargs="+", type=int, default=[126, 127, 128, 129, 130]
    )
    parser.add_argument("--attacker-seeds", nargs="+", type=int, default=[10125, 20125, 30125])
    parser.add_argument(
        "--knowledge",
        nargs="+",
        choices=("training", "inference"),
        default=["training", "inference"],
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--legacy-noise-std", type=float, default=0.31)
    parser.add_argument("--semantic-noise-std", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.attempts <= 0:
        raise ValueError("epochs, batch size, and attempts must be positive")
    if args.retry_seconds < 0:
        raise ValueError("retry seconds must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not (args.data_root / "train").is_dir() or not (args.data_root / "val").is_dir():
        raise FileNotFoundError(f"FaceScrub train/val split not found: {args.data_root}")

    checkpoints = {
        seed: args.targets_root / f"seed{seed}" / "stage2" / "checkpoint_best.pt"
        for seed in args.target_seeds
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing target checkpoints: {missing}")

    state_path = args.state.resolve()
    state = load_state(state_path)
    state["protocol"] = {
        "target_seeds": args.target_seeds,
        "attacker_seeds": args.attacker_seeds,
        "knowledge": args.knowledge,
        "scenarios": args.scenarios,
        "epochs": args.epochs,
        "matched_width": 128,
        "matched_residual_blocks": 4,
        "legacy_noise_std": args.legacy_noise_std,
        "semantic_noise_std": args.semantic_noise_std,
        "oracle_is_upper_bound_only": True,
    }
    save_json(state_path, state)

    commands = []
    for target_seed, checkpoint in checkpoints.items():
        for attacker_seed in args.attacker_seeds:
            for knowledge in args.knowledge:
                for scenario in args.scenarios:
                    attack_view, conditioning = SCENARIOS[scenario]
                    output_dir = (
                        args.results_root
                        / f"target_seed{target_seed}"
                        / f"attacker_seed{attacker_seed}"
                        / f"{scenario}_{knowledge}"
                    )
                    name = (
                        f"identity_attack_t{target_seed}_a{attacker_seed}_"
                        f"{scenario}_{knowledge}"
                    )
                    command = [
                        sys.executable,
                        "scripts/attack_dual_path_facescrub.py",
                        "--dual-path-checkpoint",
                        str(checkpoint),
                        "--data-root",
                        str(args.data_root),
                        "--output-dir",
                        str(output_dir),
                        "--attack-type",
                        "decoder",
                        "--attack-view",
                        attack_view,
                        "--identity-conditioning",
                        conditioning,
                        "--attack-knowledge",
                        knowledge,
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--workers",
                        str(args.workers),
                        "--width",
                        "128",
                        "--residual-blocks",
                        "4",
                        "--legacy-noise-std",
                        str(args.legacy_noise_std),
                        "--semantic-noise-std",
                        str(args.semantic_noise_std),
                        "--seed",
                        str(attacker_seed),
                        "--evaluation-noise-seeds",
                        str(800_000 + target_seed),
                        str(900_000 + target_seed),
                        str(1_000_000 + target_seed),
                        "--perceptual-metrics",
                        "--device",
                        args.device,
                    ]
                    if args.legacy_checkpoint_dir is not None:
                        command.extend(
                            ("--legacy-checkpoint-dir", str(args.legacy_checkpoint_dir))
                        )
                    commands.append(command)
                    if not args.dry_run:
                        run_job(
                            state,
                            state_path,
                            name,
                            command,
                            output_dir,
                            args.log_root,
                            args.attempts,
                            args.retry_seconds,
                        )

    if args.dry_run:
        print(
            json.dumps(
                {"status": "DRY_RUN", "jobs": len(commands), "first_command": commands[0]}
            )
        )
        return

    summary_path = args.results_root / "identity_conditioned_attack_summary.json"
    aggregate_command = [
        sys.executable,
        "scripts/aggregate_identity_conditioned_attacks.py",
        "--results-root",
        str(args.results_root),
        "--target-seeds",
        *[str(seed) for seed in args.target_seeds],
        "--attacker-seeds",
        *[str(seed) for seed in args.attacker_seeds],
        "--knowledge",
        *args.knowledge,
        "--scenarios",
        *args.scenarios,
    ]
    result = subprocess.run(aggregate_command, cwd=ROOT, check=False)
    if result.returncode != 0 or not summary_path.is_file():
        raise RuntimeError("identity-conditioned attack aggregation failed")
    state["status"] = "PASS"
    state["finished_at_utc"] = now()
    state["summary"] = str(summary_path)
    save_json(state_path, state)
    print(
        json.dumps({"status": "IDENTITY_ATTACK_EVIDENCE_READY", "summary": str(summary_path)})
    )


if __name__ == "__main__":
    main()
