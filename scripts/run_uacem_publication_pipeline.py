#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from scripts.run_full_publication_pipeline import (
        PipelineStep,
        ProcessSpec,
        formal_source_fingerprint,
        load_state,
        parse_gpu_indices,
        run_step,
        save_state,
        single_step,
        validate_cuda,
    )
except ModuleNotFoundError:
    from run_full_publication_pipeline import (
        PipelineStep,
        ProcessSpec,
        formal_source_fingerprint,
        load_state,
        parse_gpu_indices,
        run_step,
        save_state,
        single_step,
        validate_cuda,
    )


ROOT = Path(__file__).resolve().parents[1]


def job_step(number: int, stage: str, gpus: tuple[int, ...]) -> PipelineStep:
    processes = []
    for shard, gpu in enumerate(gpus):
        processes.append(
            ProcessSpec(
                (
                    sys.executable,
                    "scripts/run_recoverable_jobs.py",
                    "--jobs",
                    f"generated_runbooks/uacem_formal/jobs/{stage}.json",
                    "--state",
                    f"run_state/uacem_formal/{stage}_shard{shard}.json",
                    "--log-dir",
                    f"run_logs/uacem_formal/{stage}/shard{shard}",
                    "--shard-index",
                    str(shard),
                    "--shard-count",
                    str(len(gpus)),
                ),
                gpu,
            )
        )
    return PipelineStep(f"{number:02d}_{stage}", tuple(processes))


def build_uacem_steps(gpus: tuple[int, ...]) -> tuple[PipelineStep, ...]:
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
            gpu=gpus[0],
        ),
        single_step(
            "02_generate_runbooks",
            sys.executable,
            "scripts/generate_uacem_formal_runbooks.py",
        ),
    ]
    ordered_stages = (
        "main_targets",
        "main_shadow_attacks",
        "main_calibrations",
        "main_evaluations",
        "main_attacks",
        "main_profiles",
        "generalisation_targets",
        "generalisation_shadow_attacks",
        "generalisation_calibrations",
        "generalisation_evaluations",
        "generalisation_attacks",
        "generalisation_profiles",
        "ablation_setup",
        "ablation_calibrations",
        "ablation_attacks",
        "paper_outputs",
    )
    steps.extend(
        job_step(number, stage, gpus)
        for number, stage in enumerate(ordered_stages, start=3)
    )
    return tuple(steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=parse_gpu_indices, default=(0,))
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "run_state/uacem_publication_pipeline.json",
    )
    parser.add_argument("--stage-retries", type=int, default=100)
    parser.add_argument("--retry-seconds", type=float, default=300.0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    steps = build_uacem_steps(args.gpus)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "steps": [step.name for step in steps],
                    "gpus": list(args.gpus),
                },
                indent=2,
            )
        )
        return
    if args.stage_retries <= 0 or args.retry_seconds < 0:
        raise ValueError("invalid stage retry configuration")
    hardware = validate_cuda(args.gpus)
    state_path = args.state.resolve()
    state = load_state(state_path)
    if state.get("root") != str(ROOT.resolve()):
        raise RuntimeError("pipeline state belongs to a different checkout")
    source_fingerprint = formal_source_fingerprint(ROOT)
    prior_fingerprint = state.get("formal_source_fingerprint")
    if prior_fingerprint is not None and prior_fingerprint != source_fingerprint:
        raise RuntimeError("formal source changed after the UA-CEM campaign started")
    state["hardware"] = hardware
    state["formal_source_fingerprint"] = source_fingerprint
    save_state(state_path, state)
    for step in steps:
        for attempt in range(1, args.stage_retries + 1):
            try:
                run_step(step, state_path, state)
                break
            except SystemExit:
                if attempt == args.stage_retries:
                    raise
                print(
                    json.dumps(
                        {
                            "step": step.name,
                            "status": "RETRY_PENDING",
                            "attempt": attempt,
                            "sleep_seconds": args.retry_seconds,
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.retry_seconds)
    print(
        json.dumps(
            {
                "status": "EXPERIMENT_EVIDENCE_READY",
                "bundle": "handoff/uacem_paper_writing_bundle.tar.gz",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
