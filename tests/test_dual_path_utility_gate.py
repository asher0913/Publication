from pathlib import Path

from scripts.extract_dual_path_utility_gate import build_gate, extract_candidate


def candidate(legacy: float, semantic: float, minimum: float) -> dict:
    return {
        "legacy_noise_std": legacy,
        "semantic_noise_std": semantic,
        "minimum_fused_accuracy": minimum,
        "mean_fused_accuracy": minimum + 0.01,
        "maximum_fused_accuracy": minimum + 0.02,
        "utility_runs": [],
    }


def test_extracts_exact_joint_noise_configuration() -> None:
    expected = candidate(0.3, 0.1, 0.82)
    probe = {"candidates": [candidate(0.25, 0.1, 0.84), expected]}
    assert extract_candidate(probe, 0.3, 0.1) is expected


def test_gate_uses_minimum_repeated_accuracy() -> None:
    result = build_gate(candidate(0.3, 0.1, 0.81), 0.8033, Path("probe.json"))
    assert result["status"] == "PASS"
    assert result["best_validation_accuracy"] == 0.81


def test_gate_rejects_mean_only_pass() -> None:
    result = build_gate(candidate(0.3, 0.1, 0.80), 0.8033, Path("probe.json"))
    assert result["status"] == "FAIL"
