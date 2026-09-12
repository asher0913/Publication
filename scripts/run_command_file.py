#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_id(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "commands": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def bind_state_to_working_directory(state: dict, working_directory: Path) -> None:
    resolved = str(working_directory.resolve())
    existing = state.get("working_directory")
    if existing is not None and existing != resolved:
        raise ValueError(
            "state file belongs to a different working directory; use a new state file"
        )
    state["working_directory"] = resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    all_commands = [
        line.strip()
        for line in args.commands.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    commands = [
        command
        for index, command in enumerate(all_commands)
        if index % args.shard_count == args.shard_index
    ]
    state = load_state(args.state)
    bind_state_to_working_directory(state, Path.cwd())
    save_state(args.state, state)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for command in commands:
        identifier = command_id(command)
        prior = state["commands"].get(identifier, {})
        if prior.get("status") == "PASS" and prior.get("command") == command:
            skipped += 1
            continue
        started = datetime.now(timezone.utc).isoformat()
        log_path = args.log_dir / f"{identifier}.log"
        running_record = {
            "command": command,
            "status": "RUNNING",
            "started_at_utc": started,
            "log": str(log_path),
        }
        state["commands"][identifier] = running_record
        save_state(args.state, state)
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    shlex.split(command),
                    text=True,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                returncode = process.wait()
        except OSError as exc:
            log_path.write_text(f"failed to start command: {exc}\n", encoding="utf-8")
            returncode = 127
        record = {
            "command": command,
            "status": "PASS" if returncode == 0 else "FAIL",
            "returncode": returncode,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "log": str(log_path),
        }
        state["commands"][identifier] = record
        save_state(args.state, state)
        print(json.dumps(record, sort_keys=True), flush=True)
        if returncode != 0:
            raise SystemExit(returncode)
        completed += 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_commands": len(commands),
                "completed_now": completed,
                "skipped_completed": skipped,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
