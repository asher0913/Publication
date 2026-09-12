import json
import sys
from pathlib import Path

import pytest

from scripts.run_command_file import (
    bind_state_to_working_directory,
    command_id,
    load_state,
    main,
)


def test_command_ids_are_stable_and_state_defaults_are_empty(tmp_path):
    assert command_id("python train.py") == command_id("python train.py")
    assert command_id("python train.py") != command_id("python test.py")
    assert load_state(tmp_path / "missing.json") == {
        "schema_version": 1,
        "commands": {},
    }


def test_command_state_cannot_be_reused_from_another_checkout(tmp_path):
    state = {"schema_version": 1, "commands": {}}
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    bind_state_to_working_directory(state, first)
    with pytest.raises(ValueError, match="different working directory"):
        bind_state_to_working_directory(state, second)


def test_command_runner_streams_log_and_records_pass(monkeypatch, tmp_path):
    commands = tmp_path / "commands.txt"
    state = tmp_path / "state.json"
    logs = tmp_path / "logs"
    commands.write_text(
        f'{sys.executable} -c "print(\'streamed output\')"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_command_file.py",
            "--commands",
            str(commands),
            "--state",
            str(state),
            "--log-dir",
            str(logs),
        ],
    )
    main()
    payload = json.loads(state.read_text(encoding="utf-8"))
    record = next(iter(payload["commands"].values()))
    assert record["status"] == "PASS"
    assert "streamed output" in Path(record["log"]).read_text(encoding="utf-8")
