#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_ready() -> tuple[bool, dict]:
    audit_path = ROOT / "results/uacem_formal/audit.json"
    bundle_path = ROOT / "handoff/uacem_paper_writing_bundle.tar.gz"
    manifest_path = ROOT / "handoff/uacem_paper_writing_bundle.json"
    audit = load_json(audit_path)
    manifest = load_json(manifest_path)
    checks = {
        "audit_pass": audit.get("status") == "PASS",
        "bundle_exists": bundle_path.is_file() and bundle_path.stat().st_size > 0,
        "manifest_ready": manifest.get("status") == "EXPERIMENT_EVIDENCE_READY",
        "bundle_hash_matches": False,
    }
    if checks["bundle_exists"] and checks["manifest_ready"]:
        checks["bundle_hash_matches"] = manifest.get("sha256") == sha256(bundle_path)
    return all(checks.values()), checks


def current_pipeline_step(pipeline_state: Path) -> str | None:
    state = load_json(pipeline_state)
    steps = state.get("steps", {})
    for name, record in reversed(list(steps.items())):
        if record.get("status") != "PASS":
            return f"{name}:{record.get('status', 'UNKNOWN')}"
    return next(reversed(steps), None) if steps else None


def latest_log_progress() -> tuple[float, str | None]:
    newest_time = 0.0
    newest_path = None
    roots = (
        ROOT / "run_logs/full_pipeline",
        ROOT / "run_logs/uacem_formal",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if modified > newest_time:
                newest_time = modified
                newest_path = str(path.relative_to(ROOT))
    return newest_time, newest_path


def stop_process_group(process: subprocess.Popen, grace_seconds: float = 60.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(1.0)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def acquire_singleton(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another UA-CEM watchdog already holds the lock") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    return stream


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the frozen UA-CEM campaign running until audited evidence exists."
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--pipeline-state",
        type=Path,
        default=ROOT / "run_state/uacem_publication_pipeline.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "run_state/uacem_watchdog.json",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--max-retry-seconds", type=float, default=900.0)
    parser.add_argument("--stall-seconds", type=float, default=7200.0)
    parser.add_argument("--minimum-free-gib", type=float, default=50.0)
    args = parser.parse_args()
    if min(
        args.poll_seconds,
        args.retry_seconds,
        args.max_retry_seconds,
        args.stall_seconds,
        args.minimum_free_gib,
    ) <= 0:
        raise ValueError("watchdog timing and disk thresholds must be positive")

    _lock = acquire_singleton(ROOT / "run_state/uacem_watchdog.lock")
    state = load_json(args.state)
    state.setdefault("schema_version", 1)
    state.setdefault("attempts", 0)
    state["root"] = str(ROOT)
    state["watchdog_pid"] = os.getpid()
    state["started_at_utc"] = state.get("started_at_utc", utc_now())

    ready, checks = evidence_ready()
    if ready:
        state.update(status="COMPLETE", completion_checks=checks, finished_at_utc=utc_now())
        save_json(args.state, state)
        print(json.dumps(state, sort_keys=True), flush=True)
        return

    retry_delay = args.retry_seconds
    console_path = ROOT / "run_logs/uacem_watchdog_console.log"
    console_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state["attempts"] += 1
        attempt = state["attempts"]
        command = [
            sys.executable,
            "scripts/run_uacem_publication_pipeline.py",
            "--gpus",
            args.gpus,
            "--state",
            str(args.pipeline_state),
            "--stage-retries",
            "1",
            "--retry-seconds",
            "0",
        ]
        with console_path.open("a", encoding="utf-8") as console:
            console.write(
                json.dumps(
                    {"event": "PIPELINE_START", "attempt": attempt, "at_utc": utc_now()}
                )
                + "\n"
            )
            console.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
                text=True,
            )
            launched = time.monotonic()
            termination_reason = None
            while process.poll() is None:
                now = time.time()
                progress_time, progress_path = latest_log_progress()
                free_gib = shutil.disk_usage(ROOT).free / (1024**3)
                state.update(
                    status="RUNNING",
                    heartbeat_at_utc=utc_now(),
                    pipeline_pid=process.pid,
                    pipeline_attempt=attempt,
                    current_step=current_pipeline_step(args.pipeline_state),
                    latest_log=progress_path,
                    latest_log_age_seconds=(now - progress_time if progress_time else None),
                    free_disk_gib=free_gib,
                )
                save_json(args.state, state)
                if free_gib < args.minimum_free_gib:
                    termination_reason = "LOW_DISK"
                    stop_process_group(process)
                    break
                if (
                    progress_time
                    and time.monotonic() - launched > args.stall_seconds
                    and now - progress_time > args.stall_seconds
                ):
                    termination_reason = "NO_LOG_PROGRESS"
                    stop_process_group(process)
                    break
                time.sleep(args.poll_seconds)

            returncode = process.returncode
            ready, checks = evidence_ready()
            console.write(
                json.dumps(
                    {
                        "event": "PIPELINE_EXIT",
                        "attempt": attempt,
                        "at_utc": utc_now(),
                        "returncode": returncode,
                        "termination_reason": termination_reason,
                        "evidence_ready": ready,
                    }
                )
                + "\n"
            )
            console.flush()

        if returncode == 0 and ready:
            completion = {
                "schema_version": 1,
                "status": "EXPERIMENT_EVIDENCE_READY",
                "finished_at_utc": utc_now(),
                "audit": "results/uacem_formal/audit.json",
                "bundle": "handoff/uacem_paper_writing_bundle.tar.gz",
                "bundle_sha256": sha256(
                    ROOT / "handoff/uacem_paper_writing_bundle.tar.gz"
                ),
                "checks": checks,
            }
            save_json(ROOT / "handoff/UACEM_EXPERIMENT_EVIDENCE_READY.json", completion)
            state.update(
                status="COMPLETE",
                finished_at_utc=completion["finished_at_utc"],
                completion_checks=checks,
                completion_marker="handoff/UACEM_EXPERIMENT_EVIDENCE_READY.json",
                pipeline_returncode=returncode,
            )
            save_json(args.state, state)
            print(json.dumps(completion, sort_keys=True), flush=True)
            return

        state.update(
            status="RETRY_PENDING",
            heartbeat_at_utc=utc_now(),
            pipeline_returncode=returncode,
            termination_reason=termination_reason,
            completion_checks=checks,
            retry_after_seconds=retry_delay,
        )
        save_json(args.state, state)
        time.sleep(retry_delay)
        retry_delay = min(args.max_retry_seconds, retry_delay * 2.0)


if __name__ == "__main__":
    main()
