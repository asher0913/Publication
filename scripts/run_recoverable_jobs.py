#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RESET_ROOTS = tuple(
    path.resolve()
    for path in (
        ROOT / "results/uacem_formal",
        ROOT / "paper/generated/uacem",
        ROOT / "handoff",
    )
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def safe_reset(path_text: str) -> None:
    path = (ROOT / path_text).resolve()
    if not any(path == root or root in path.parents for root in ALLOWED_RESET_ROOTS):
        raise ValueError(f"refusing to reset path outside formal results: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def completed(job: dict) -> bool:
    path = ROOT / job["completion"]
    if not path.is_file():
        return False
    requirements = job.get("completion_json")
    if requirements is None:
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(payload.get(key) == value for key, value in requirements.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    if args.attempts <= 0 or args.retry_seconds < 0:
        raise ValueError("invalid retry configuration")

    all_jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    jobs = [
        job for index, job in enumerate(all_jobs)
        if index % args.shard_count == args.shard_index
    ]
    state = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.is_file()
        else {"schema_version": 1, "jobs": {}}
    )
    state["working_directory"] = str(ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.state, state)
    unresolved = []
    for job_spec in jobs:
        name = job_spec["name"]
        if completed(job_spec):
            state["jobs"][name] = {
                "status": "PASS",
                "completion": job_spec["completion"],
                "recovered_from_artifact": True,
            }
            save_json(args.state, state)
            continue
        returncode = 1
        prior_attempts = int(state["jobs"].get(name, {}).get("total_attempts", 0))
        for local_attempt in range(1, args.attempts + 1):
            attempt = prior_attempts + local_attempt
            for path in job_spec.get("reset_paths", []):
                safe_reset(path)
            log_path = args.log_dir / f"{name}.attempt{attempt}.log"
            started = datetime.now(timezone.utc).isoformat()
            state["jobs"][name] = {
                "status": "RUNNING",
                "attempt": attempt,
                "total_attempts": attempt,
                "command": job_spec["command"],
                "started_at_utc": started,
                "log": str(log_path),
            }
            save_json(args.state, state)
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    job_spec["command"],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    start_new_session=False,
                )
                while True:
                    try:
                        returncode = int(process.wait(timeout=60.0))
                        break
                    except subprocess.TimeoutExpired:
                        state["jobs"][name].update(
                            heartbeat_at_utc=datetime.now(timezone.utc).isoformat(),
                            pid=process.pid,
                            log_bytes=log_path.stat().st_size if log_path.exists() else 0,
                        )
                        save_json(args.state, state)
            if returncode == 0 and completed(job_spec):
                break
            if local_attempt < args.attempts:
                time.sleep(args.retry_seconds)
        status = "PASS" if returncode == 0 and completed(job_spec) else "FAIL"
        state["jobs"][name] = {
            **state["jobs"][name],
            "status": status,
            "returncode": returncode,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "completion": job_spec["completion"],
        }
        save_json(args.state, state)
        print(json.dumps({"job": name, "status": status}), flush=True)
        if status == "FAIL":
            unresolved.append(name)
    if unresolved:
        raise SystemExit(f"unresolved jobs: {unresolved}")
    print(json.dumps({"status": "PASS", "jobs": len(jobs)}, sort_keys=True))


if __name__ == "__main__":
    main()
