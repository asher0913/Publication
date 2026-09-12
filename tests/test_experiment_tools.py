from scripts.aggregate_results import bootstrap_interval, method_name
import shlex
from pathlib import Path

from scripts.expand_attack_matrix import build_commands, build_protocol_commands
from scripts.expand_matrix import expand_matrix
from scripts.generate_runbooks import (
    freeze_generalisation_matrix,
    generalisation_attack_commands,
)
from scripts.summarise_formal_pilot import summarise


def test_attack_matrix_expands_selected_targets_and_seeds():
    matrix = {
        "results_root": "results/core",
        "experiments": [
            {"name": "seed125_slots8", "overrides": {}},
            {"name": "seed125_slots8_weight003", "overrides": {}},
        ],
    }
    commands = build_commands(
        matrix,
        [101, 202],
        r"seed125_slots8",
        ["conv_decoder", "adaptive"],
    )
    assert len(commands) == 4
    assert "checkpoint_best.pt" in commands[0]
    assert "attacks/conv_decoder/seed202" in commands[1]
    assert "--attack-type adaptive" in commands[2]


def test_generated_target_command_preserves_config_paths_with_spaces(tmp_path):
    output_dir = tmp_path / "directory with spaces"
    commands = expand_matrix(
        {"output_dir": "unused"},
        {
            "results_root": "results/test",
            "experiments": [{"name": "run", "overrides": {}}],
        },
        output_dir,
        selection_only=True,
    )
    arguments = shlex.split(commands[0])
    config_path = Path(arguments[arguments.index("--config") + 1])
    assert config_path == output_dir / "run.json"
    assert config_path.is_file()


def test_aggregation_distinguishes_privacy_weights_and_bootstraps_deterministically():
    base = {
        "defense": {
            "name": "prototype_cem",
            "privacy_weight": 0.1,
            "noise_std": 0.025,
            "regularizer": {"num_slots": 8},
        }
    }
    lower, upper = bootstrap_interval([1.0, 2.0, 3.0], iterations=200)
    assert lower <= 2.0 <= upper
    assert method_name(base) == "prototype_cem_k8_w0.1_n0.025"

    uacem = {
        "defense": {
            "name": "official_cem",
            "noise_std": 0.025,
            "variant": "utility_aligned_anisotropic_cem",
            "calibration": {
                "score": "reconstruction_to_task",
                "alpha": 0.3,
            },
        }
    }
    assert method_name(uacem) == "uacem_reconstruction_to_task_a0.3_n0.025"


def test_tiered_attack_protocol_avoids_duplicate_expensive_attacks():
    matrix = {
        "results_root": "results/core",
        "experiments": [
            {"name": "seed125_official_cem", "overrides": {}},
            {"name": "seed125_slots8", "overrides": {}},
            {"name": "seed125_gaussian", "overrides": {}},
        ],
    }
    protocol = {
        "attacker_seeds": [101, 202],
        "tiers": [
            {
                "target_name_regex": "seed125_.*",
                "attack_types": ["conv_decoder"],
            },
            {
                "target_name_regex": "seed125_(official_cem|slots8)",
                "attack_types": ["gan", "adaptive"],
            },
        ],
    }
    commands = build_protocol_commands(matrix, protocol)
    assert len(commands) == 14
    assert len(commands) == len(set(commands))


def test_generalisation_transfers_test_blind_settings_and_generates_attacks():
    matrix = {
        "results_root": "results/generalisation",
        "experiments": [
            {
                "name": f"run_{index}_official" if index % 2 else f"run_{index}_proposed",
                "overrides": (
                    {"defense.name": "official_cem"} if index % 2 else {}
                ),
            }
            for index in range(30)
        ],
    }
    selection = {
        "selected": {
            "official_cem": {"noise_std": 0.015, "privacy_weight": 0.03},
            "proposed": {"noise_std": 0.025, "privacy_weight": 0.1},
        }
    }
    frozen = freeze_generalisation_matrix(matrix, selection)
    assert frozen["frozen_from_test_blind_selection"] is True
    assert frozen["experiments"][0]["overrides"]["defense.noise_std"] == 0.025
    assert frozen["experiments"][1]["overrides"]["defense.noise_std"] == 0.015
    protocol = {
        "attacker_seeds": [10125, 20125, 30125],
        "attack_types": ["conv_decoder"],
        "expected_attack_runs": 90,
    }
    commands = generalisation_attack_commands(frozen, protocol)
    assert len(commands) == 90
    assert len(commands) == len(set(commands))


def test_formal_pilot_summary_requires_all_methods_and_no_test_access(tmp_path):
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        '{"cuda_available": true, "gpus": [{"name": "test-gpu"}]}',
        encoding="utf-8",
    )
    methods = {
        "seed119_no_defense": ("none", ("conv_decoder",)),
        "seed119_official_cem": ("official_cem", ("conv_decoder",)),
        "seed119_proposed": (
            "prototype_cem",
            ("conv_decoder", "residual_decoder", "gan", "adaptive"),
        ),
    }
    for run_name, (defense, attack_types) in methods.items():
        run_dir = tmp_path / "results" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(
            '{"defense": {"name": "' + defense + '"}}', encoding="utf-8"
        )
        (run_dir / "training.jsonl").write_text(
            "\n".join(
                '{"total_epoch_seconds": 2.0, "peak_gpu_memory_bytes": 1024}'
                for _ in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "target_selection_metrics.json").write_text(
            '{"target_test_accessed": false, "best_validation_accuracy": 0.8}',
            encoding="utf-8",
        )
        (run_dir / "checkpoint_best.pt").write_bytes(b"checkpoint")
        for attack_type in attack_types:
            attack_dir = run_dir / "pilot_attacks" / attack_type
            attack_dir.mkdir(parents=True)
            (attack_dir / "attack_training.jsonl").write_text(
                "\n".join('{"epoch_seconds": 1.0}' for _ in range(2)) + "\n",
                encoding="utf-8",
            )
            (attack_dir / "attack_selection_metrics.json").write_text(
                '{"target_test_accessed": false, "best_auxiliary_validation_mse": 0.1}',
                encoding="utf-8",
            )
            (attack_dir / "decoder.pt").write_bytes(b"decoder")
    payload = summarise(tmp_path / "results", environment_path)
    assert payload["status"] == "PASS"
    assert payload["target_test_accessed"] is False
    assert set(payload["methods"]) == set(methods)
