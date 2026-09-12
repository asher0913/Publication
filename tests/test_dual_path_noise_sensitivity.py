from scripts.evaluate_dual_path_noise_sensitivity import aggregate


def test_noise_aggregation_keeps_operating_points_separate() -> None:
    rows = [
        {"legacy_noise_std": 0.22, "semantic_noise_std": 0.10, "fused_accuracy": 0.84},
        {"legacy_noise_std": 0.22, "semantic_noise_std": 0.10, "fused_accuracy": 0.82},
        {"legacy_noise_std": 0.31, "semantic_noise_std": 0.10, "fused_accuracy": 0.81},
    ]
    points = aggregate(rows)
    assert len(points) == 2
    assert points[0]["runs"] == 2
    assert abs(points[0]["mean_fused_accuracy"] - 0.83) < 1e-12
    assert points[1]["minimum_fused_accuracy"] == 0.81
