from scripts.attack_dual_path_facescrub import resolve_noise_std
from scripts.calibrate_dual_path_noise import select_strongest_passing_noise


def test_selects_largest_noise_with_conservative_utility_pass() -> None:
    candidates = [
        {"semantic_noise_std": 0.1, "minimum_fused_accuracy": 0.84},
        {"semantic_noise_std": 0.3, "minimum_fused_accuracy": 0.81},
        {"semantic_noise_std": 0.5, "minimum_fused_accuracy": 0.79},
    ]
    selected = select_strongest_passing_noise(candidates, 0.8033, 0.002)
    assert selected is not None
    assert selected["semantic_noise_std"] == 0.3


def test_returns_none_when_no_noise_passes_utility() -> None:
    candidates = [
        {"semantic_noise_std": 0.1, "minimum_fused_accuracy": 0.80},
    ]
    assert select_strongest_passing_noise(candidates, 0.8033, 0.0) is None


def test_attack_noise_override_is_explicit() -> None:
    assert resolve_noise_std(0.05, None) == 0.05
    assert resolve_noise_std(0.05, 0.2) == 0.2
