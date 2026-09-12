#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProcessSpec:
    command: tuple[str, ...]
    gpu: int | None = None


@dataclass(frozen=True)
class PipelineStep:
    name: str
    processes: tuple[ProcessSpec, ...]


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("at least one non-negative GPU index is required")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return indices


def sharded_runner_step(
    name: str, command_file: str, gpus: tuple[int, ...]
) -> PipelineStep:
    shard_count = len(gpus)
    processes = []
    for shard_index, gpu in enumerate(gpus):
        processes.append(
            ProcessSpec(
                (
                    sys.executable,
                    "scripts/run_command_file.py",
                    "--commands",
                    command_file,
                    "--state",
                    f"run_state/full/{name}_shard{shard_index}.json",
                    "--log-dir",
                    f"run_logs/{name}/shard{shard_index}",
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(shard_count),
                ),
                gpu,
            )
        )
    return PipelineStep(name, tuple(processes))


def single_step(name: str, *command: str, gpu: int | None = None) -> PipelineStep:
    return PipelineStep(name, (ProcessSpec(tuple(command), gpu),))


def build_pipeline_steps(gpus: tuple[int, ...]) -> tuple[PipelineStep, ...]:
    first_gpu = gpus[0]
    steps = [
        single_step(
            "01_preflight",
            sys.executable,
            "scripts/preflight_formal_run.py",
            "--prepare-assets",
            "--device",
            "cuda",
            "--gate",
            "code",
            gpu=first_gpu,
        ),
        single_step(
            "02_generate_pilot",
            sys.executable,
            "scripts/generate_runbooks.py",
            "--stage",
            "pilot",
        ),
        sharded_runner_step(
            "03_pilot_targets",
            "generated_runbooks/pilot/01_pilot_targets.txt",
            gpus,
        ),
        sharded_runner_step(
            "04_pilot_attacks",
            "generated_runbooks/pilot/02_pilot_attacks.txt",
            gpus,
        ),
        single_step(
            "05_pilot_gate",
            sys.executable,
            "scripts/run_command_file.py",
            "--commands",
            "generated_runbooks/pilot/03_pilot_audit.txt",
            "--state",
            "run_state/full/pilot_audit.json",
            "--log-dir",
            "run_logs/pilot_audit",
            gpu=first_gpu,
        ),
        single_step(
            "06_generate_selection",
            sys.executable,
            "scripts/generate_runbooks.py",
            "--stage",
            "selection",
        ),
        sharded_runner_step(
            "07_selection_targets",
            "generated_runbooks/selection/01_targets.txt",
            gpus,
        ),
        sharded_runner_step(
            "08_selection_attacks",
            "generated_runbooks/selection/02_selection_attacks.txt",
            gpus,
        ),
        single_step(
            "09_freeze_selection",
            sys.executable,
            "scripts/run_command_file.py",
            "--commands",
            "generated_runbooks/selection/03_freeze.txt",
            "--state",
            "run_state/full/selection_freeze.json",
            "--log-dir",
            "run_logs/selection_freeze",
        ),
        single_step(
            "10_pre_run_gate",
            sys.executable,
            "scripts/preflight_formal_run.py",
            "--gate",
            "pre_run",
            gpu=first_gpu,
        ),
        single_step(
            "11_generate_headline",
            sys.executable,
            "scripts/generate_runbooks.py",
            "--stage",
            "headline",
        ),
        sharded_runner_step(
            "12_headline_targets",
            "generated_runbooks/headline/01_headline_targets.txt",
            gpus,
        ),
        sharded_runner_step(
            "13_headline_attacks",
            "generated_runbooks/headline/02_headline_attacks.txt",
            gpus,
        ),
        sharded_runner_step(
            "14_headline_profiles",
            "generated_runbooks/headline/03_headline_profiles.txt",
            gpus,
        ),
    ]
    next_number = 15
    for dataset in ("facescrub", "cifar100", "cifar10"):
        steps.append(
            sharded_runner_step(
                f"{next_number:02d}_{dataset}_generalisation_targets",
                f"generated_runbooks/headline/04_{dataset}_generalisation_targets.txt",
                gpus,
            )
        )
        next_number += 1
        steps.append(
            sharded_runner_step(
                f"{next_number:02d}_{dataset}_generalisation_attacks",
                f"generated_runbooks/headline/04_{dataset}_generalisation_attacks.txt",
                gpus,
            )
        )
        next_number += 1
    steps.extend(
        [
            single_step(
                "21_aggregate_and_generate_assets",
                sys.executable,
                "scripts/run_command_file.py",
                "--commands",
                "generated_runbooks/headline/05_aggregate_analyse.txt",
                "--state",
                "run_state/full/aggregate_analyse.json",
                "--log-dir",
                "run_logs/aggregate_analyse",
            ),
            single_step(
                "22_final_evidence_audit",
                sys.executable,
                "scripts/run_command_file.py",
                "--commands",
                "generated_runbooks/headline/06_final_audit.txt",
                "--state",
                "run_state/full/final_audit.json",
                "--log-dir",
                "run_logs/final_audit",
            ),
            single_step(
                "23_build_paper_handoff",
                sys.executable,
                "scripts/build_paper_handoff.py",
            ),
        ]
    )
    return tuple(steps)


def validate_cuda(gpus: tuple[int, ...]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the formal pipeline requires NVIDIA CUDA")
    count = torch.cuda.device_count()
    invalid = [index for index in gpus if index >= count]
    if invalid:
        raise RuntimeError(f"GPU indices {invalid} are outside visible device count {count}")
    names = [torch.cuda.get_device_name(index) for index in gpus]
    if len(set(names)) != 1:
        raise RuntimeError(
            "all selected GPUs must have the same model for comparable timing: "
            + ", ".join(f"{index}={name}" for index, name in zip(gpus, names))
        )
    return {"visible_device_count": count, "selected_gpus": list(gpus), "models": names}


def step_fingerprint(step: PipelineStep) -> str:
    payload = [
        {"command": list(process.command), "gpu": process.gpu}
        for process in step.processes
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def formal_source_fingerprint(root: Path) -> str:
    included = []
    for directory in ("src", "scripts", "configs", "environment"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.name == "frozen_facescrub_core_matrix.json":
                continue
            included.append(path)
    included.append(root / "pyproject.toml")
    digest = hashlib.sha256()
    for path in included:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "root": str(ROOT.resolve()), "steps": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run_step(step: PipelineStep, state_path: Path, state: dict) -> None:
    fingerprint = step_fingerprint(step)
    prior = state["steps"].get(step.name, {})
    if prior.get("status") == "PASS" and prior.get("fingerprint") == fingerprint:
        print(json.dumps({"step": step.name, "status": "SKIP_ALREADY_PASS"}))
        return

    started = datetime.now(timezone.utc).isoformat()
    log_dir = ROOT / "run_logs" / "full_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, process in enumerate(step.processes):
        records.append(
            {
                "command": list(process.command),
                "gpu": process.gpu,
                "log": str(log_dir / f"{step.name}_process{index}.log"),
            }
        )
    state["steps"][step.name] = {
        "status": "RUNNING",
        "fingerprint": fingerprint,
        "started_at_utc": started,
        "processes": records,
    }
    save_state(state_path, state)
    print(json.dumps({"step": step.name, "status": "RUNNING"}), flush=True)

    running = []
    streams = []
    try:
        for process, record in zip(step.processes, records):
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            if process.gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(process.gpu)
            stream = Path(record["log"]).open("w", encoding="utf-8")
            streams.append(stream)
            running.append(
                subprocess.Popen(
                    list(process.command),
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    text=True,
                    start_new_session=True,
                )
            )

        failed_returncode = None
        while running:
            statuses = [process.poll() for process in running]
            failed = [code for code in statuses if code not in (None, 0)]
            if failed:
                failed_returncode = failed[0]
                for process in running:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                for process in running:
                    process.wait()
                break
            if all(code == 0 for code in statuses):
                break
            time.sleep(1.0)
    except BaseException:
        for process in running:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in running:
            process.wait()
        raise
    finally:
        for stream in streams:
            stream.close()

    returncodes = [process.returncode for process in running]
    status = "PASS" if all(code == 0 for code in returncodes) else "FAIL"
    state["steps"][step.name] = {
        "status": status,
        "fingerprint": fingerprint,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncodes": returncodes,
        "processes": records,
    }
    save_state(state_path, state)
    print(json.dumps({"step": step.name, "status": status, "returncodes": returncodes}))
    if status != "PASS":
        raise SystemExit(failed_returncode or 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen publication experiment campaign from preflight to handoff."
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_indices,
        default=(0,),
        help="comma-separated visible GPU indices; selected GPUs must be the same model",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "run_state/full_publication_pipeline.json",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the stage plan without requiring CUDA or running commands",
    )
    args = parser.parse_args()
    steps = build_pipeline_steps(args.gpus)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "gpus": list(args.gpus),
                    "steps": [
                        {
                            "name": step.name,
                            "processes": [list(spec.command) for spec in step.processes],
                        }
                        for step in steps
                    ],
                },
                indent=2,
            )
        )
        return

    hardware = validate_cuda(args.gpus)
    state_path = args.state.resolve()
    state = load_state(state_path)
    if state.get("root") != str(ROOT.resolve()):
        raise RuntimeError("pipeline state belongs to a different Publication checkout")
    previous_hardware = state.get("hardware")
    if previous_hardware is not None and previous_hardware != hardware:
        raise RuntimeError("GPU selection changed after the formal pipeline started")
    source_fingerprint = formal_source_fingerprint(ROOT)
    previous_source_fingerprint = state.get("formal_source_fingerprint")
    if (
        previous_source_fingerprint is not None
        and previous_source_fingerprint != source_fingerprint
    ):
        raise RuntimeError("formal code/configuration changed after the pipeline started")
    state["hardware"] = hardware
    state["formal_source_fingerprint"] = source_fingerprint
    save_state(state_path, state)
    for step in steps:
        run_step(step, state_path, state)
    print(
        json.dumps(
            {
                "status": "EXPERIMENT_EVIDENCE_READY",
                "handoff": "handoff/paper_writing_bundle.tar.gz",
                "state": str(state_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
