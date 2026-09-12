#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run(*command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, check=check, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install and start the resumable UA-CEM user service."
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--unit", default="uacem-publication.service")
    parser.add_argument("--enable-linger", action="store_true")
    args = parser.parse_args()

    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / args.unit
    command = " ".join(
        quote(str(part))
        for part in (
            python,
            ROOT / "scripts/run_uacem_watchdog.py",
            "--gpus",
            args.gpus,
        )
    )
    unit = f"""[Unit]
Description=UA-CEM formal publication experiment
After=default.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={ROOT}
ExecStart={command}
Restart=on-failure
RestartSec=30s
TimeoutStopSec=120s
KillMode=control-group
Environment=PYTHONUNBUFFERED=1
Environment=PATH={python.parent}:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
"""
    unit_path.write_text(unit, encoding="utf-8")
    if args.enable_linger:
        run("loginctl", "enable-linger", os.environ.get("USER", ""), check=False)
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "enable", "--now", args.unit)
    status = run(
        "systemctl", "--user", "show", args.unit,
        "--property=ActiveState,SubState,MainPID,Result",
    )
    print(status.stdout.strip())
    print(f"unit={unit_path}")


if __name__ == "__main__":
    main()
