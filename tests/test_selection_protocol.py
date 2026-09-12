from scripts.expand_selection_attacks import build_selection_commands
from scripts.freeze_headline_matrix import build_frozen_matrix
from scripts.select_hyperparameters import choose


def test_selection_commands_are_explicitly_test_blind():
    matrix = {
        "results_root": "results/selection",
        "experiments": [{"name": "candidate", "overrides": {}}],
    }
    commands = build_selection_commands(matrix, 10120)
    assert len(commands) == 1
    assert "--selection-only" in commands[0]


def test_selection_and_freezing_use_independent_headline_seeds():
    candidates = [
        {"family": "none", "validation_accuracy": 0.80},
        {
            "family": "gaussian",
            "validation_accuracy": 0.795,
            "attacker_validation_mse": 0.02,
            "noise_std": 0.025,
            "privacy_weight": 0.0,
        },
        {
            "family": "official_cem",
            "validation_accuracy": 0.795,
            "attacker_validation_mse": 0.03,
            "noise_std": 0.025,
            "privacy_weight": 0.1,
        },
        {
            "family": "proposed",
            "validation_accuracy": 0.796,
            "attacker_validation_mse": 0.04,
            "noise_std": 0.025,
            "privacy_weight": 0.1,
        },
    ]
    selection = choose(candidates, 0.01)
    matrix = build_frozen_matrix(selection)
    assert len(matrix["experiments"]) == 25
    assert {item["overrides"]["seed"] for item in matrix["experiments"]} == set(
        range(125, 130)
    )
