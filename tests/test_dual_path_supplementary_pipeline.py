import subprocess
import sys
from pathlib import Path

import scripts.run_dual_path_supplementary_pipeline as pipeline
from publication_cem.semantic_backbones import checkpoint_backbone_name
from scripts.aggregate_dual_path_supplementary import validate_capacity_match


ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_script_supports_direct_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_dual_path_supplementary.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_supplementary_protocol_uses_five_capacity_and_three_backbone_targets() -> None:
    assert pipeline.CAPACITY_SEEDS == (126, 127, 128, 129, 130)
    assert pipeline.BACKBONE_SEEDS == (126, 127, 128)
    assert pipeline.ATTACKER_SEEDS == (10125, 20125, 30125)


def test_capacity_control_command_matches_epoch_budget(monkeypatch, tmp_path) -> None:
    commands = {}

    def capture(
        state,
        state_path,
        name,
        command,
        completion,
        reset,
        attempts,
        retry_seconds,
    ) -> None:
        commands[name] = command

    monkeypatch.setattr(pipeline, "run_job", capture)
    monkeypatch.setattr(pipeline, "RESULTS_ROOT", tmp_path / "supplementary")
    monkeypatch.setattr(pipeline, "MAIN_RESULTS_ROOT", tmp_path / "main")
    pipeline.train_capacity_control(
        {},
        tmp_path / "state.json",
        tmp_path / "data",
        tmp_path / "legacy",
        126,
        3,
        0.0,
    )
    command = commands["capacity_target_seed126"]
    assert command[command.index("--epochs") + 1] == "40"
    assert command[command.index("--legacy-noise-std") + 1] == "0.025"
    assert "--train-legacy-server" not in command
    assert checkpoint_backbone_name({}) == "mobilenet_v3_large"


def test_resnet_control_is_explicit_in_both_training_stages(
    monkeypatch, tmp_path
) -> None:
    commands = {}

    def capture(
        state,
        state_path,
        name,
        command,
        completion,
        reset,
        attempts,
        retry_seconds,
    ) -> None:
        commands[name] = command

    monkeypatch.setattr(pipeline, "run_job", capture)
    monkeypatch.setattr(pipeline, "RESULTS_ROOT", tmp_path / "supplementary")
    pipeline.train_resnet_control(
        {},
        Path("state.json"),
        Path("data"),
        Path("legacy"),
        126,
        3,
        0.0,
    )
    for name in ("resnet18_target_seed126_stage1", "resnet18_target_seed126_stage2"):
        command = commands[name]
        assert command[command.index("--semantic-backbone") + 1] == "resnet18"


def test_capacity_validation_requires_exact_parameter_and_payload_match() -> None:
    profile = {
        "target_seed": 126,
        "client_parameters": 20,
        "server_parameters": 10,
        "total_payload_elements": 4352,
        "transmitted_bytes_per_sample_fp32": 17408,
    }
    result = validate_capacity_match(
        {"profile_rows": [profile]},
        {"profile_rows": [dict(profile)]},
    )
    assert result["status"] == "PASS"


def test_final_build_command_has_one_output_argument(monkeypatch, tmp_path) -> None:
    commands = {}

    def capture(
        state,
        state_path,
        name,
        command,
        completion,
        reset,
        attempts,
        retry_seconds,
    ) -> None:
        commands[name] = command

    monkeypatch.setattr(pipeline, "run_job", capture)
    monkeypatch.setattr(pipeline, "RESULTS_ROOT", tmp_path / "supplementary")
    monkeypatch.setattr(pipeline, "MAIN_RESULTS_ROOT", tmp_path / "main")
    monkeypatch.setattr(pipeline, "PAPER_OUTPUT", tmp_path / "paper")
    pipeline.aggregate_and_build({}, tmp_path / "state.json", 3, 0.0)
    command = commands["build_pattern_recognition_paper"]
    assert command.count("--output") == 1
    assert command[command.index("--output") + 1] == str(
        tmp_path / "supplementary" / "paper_build.json"
    )
