#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "output_tail": completed.stdout.splitlines()[-20:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="download benchmark/evaluator assets and calibrate the face protocol",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--gate",
        choices=("code", "pilot", "pre_run"),
        default="code",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/formal_preflight.json",
    )
    args = parser.parse_args()
    commands = []
    if args.prepare_assets:
        commands.extend(
            [
                [sys.executable, "scripts/capture_environment.py"],
                [sys.executable, "scripts/validate_formal_environment.py"],
                [
                    sys.executable,
                    "scripts/build_torchvision_protocol.py",
                    "--dataset",
                    "cifar10",
                ],
                [
                    sys.executable,
                    "scripts/build_torchvision_protocol.py",
                    "--dataset",
                    "cifar100",
                ],
                [sys.executable, "scripts/prefetch_evaluation_assets.py"],
                [sys.executable, "scripts/build_face_identity_protocol.py"],
            ]
        )
        if args.device:
            commands[4].extend(["--device", args.device])
            commands[5].extend(["--device", args.device])
    commands.extend(
        [
            ["bash", "scripts/run_checks.sh"],
            [sys.executable, "scripts/verify_official_cem_equivalence.py"],
            [sys.executable, "scripts/smoke_publication_pipeline.py"],
            [sys.executable, "scripts/smoke_attack_suite.py"],
            [sys.executable, "scripts/smoke_facescrub_mechanism.py"],
            [sys.executable, "scripts/smoke_evaluation_pipeline.py"],
            [
                sys.executable,
                "scripts/audit_publication_readiness.py",
                "--run-checks",
                "--allow-incomplete",
            ],
        ]
    )
    records = []
    status = "PASS"
    error = None
    try:
        for command in commands:
            records.append(run(command))
    except RuntimeError as exc:
        status = "FAIL"
        error = str(exc)
    audit_path = ROOT / "results/readiness_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "code_status": audit.get("code_status"),
        "pilot_status": audit.get("pilot_status"),
        "pre_run_status": audit.get("pre_run_status"),
        "commands": records,
        "error": error,
    }
    expected = {
        "code": "CODE_READY",
        "pilot": "PILOT_READY",
        "pre_run": "PRE_RUN_READY",
    }[args.gate]
    actual = {
        "code": payload["code_status"],
        "pilot": payload["pilot_status"],
        "pre_run": payload["pre_run_status"],
    }[args.gate]
    if status == "PASS" and actual != expected:
        status = "FAIL"
        payload["status"] = status
        payload["error"] = f"{args.gate} gate failed: expected {expected}, got {actual}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "code_status",
                    "pilot_status",
                    "pre_run_status",
                )
            }
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
