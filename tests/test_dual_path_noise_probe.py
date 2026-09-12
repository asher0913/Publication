from scripts.probe_dual_path_noise_grid import select_probe_candidate


def test_selects_candidate_with_best_worst_case_privacy_margin() -> None:
    candidates = [
        {
            "probe_pass": True,
            "minimum_normalised_privacy_margin": 1.1,
            "minimum_fused_accuracy": 0.84,
        },
        {
            "probe_pass": True,
            "minimum_normalised_privacy_margin": 1.3,
            "minimum_fused_accuracy": 0.82,
        },
        {
            "probe_pass": False,
            "minimum_normalised_privacy_margin": 2.0,
            "minimum_fused_accuracy": 0.79,
        },
    ]
    assert select_probe_candidate(candidates) is candidates[1]


def test_returns_none_when_every_candidate_fails_probe() -> None:
    assert select_probe_candidate([{"probe_pass": False}]) is None


def test_probe_selection_still_requires_utility_pass() -> None:
    candidate = {
        "probe_pass": False,
        "minimum_normalised_privacy_margin": 2.0,
        "minimum_fused_accuracy": 0.7,
    }
    assert select_probe_candidate([candidate]) is None
