#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, OSError):
        return False
    return True


def main() -> None:
    watchdog = load(ROOT / "run_state/uacem_watchdog.json")
    pipeline = load(ROOT / "run_state/uacem_publication_pipeline.json")
    job_counts = Counter()
    for path in (ROOT / "run_state/uacem_formal").glob("*.json"):
        for record in load(path).get("jobs", {}).values():
            job_counts[record.get("status", "UNKNOWN")] += 1
    steps = pipeline.get("steps", {})
    current_step = None
    for name, record in reversed(list(steps.items())):
        if record.get("status") != "PASS":
            current_step = {"name": name, "status": record.get("status")}
            break
    payload = {
        "watchdog_status": watchdog.get("status", "NOT_STARTED"),
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_alive": alive(watchdog.get("watchdog_pid")),
        "pipeline_pid": watchdog.get("pipeline_pid"),
        "pipeline_alive": alive(watchdog.get("pipeline_pid")),
        "heartbeat_at_utc": watchdog.get("heartbeat_at_utc"),
        "current_step": current_step,
        "job_status_counts": dict(sorted(job_counts.items())),
        "latest_log": watchdog.get("latest_log"),
        "latest_log_age_seconds": watchdog.get("latest_log_age_seconds"),
        "free_disk_gib": watchdog.get("free_disk_gib"),
        "completion_marker_exists": (
            ROOT / "handoff/UACEM_EXPERIMENT_EVIDENCE_READY.json"
        ).is_file(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
