from scripts.statistical_analysis import paired_analysis


def test_paired_analysis_uses_target_pairs_and_predeclared_margin():
    rows = []
    for seed in range(5):
        for method, mse, accuracy in (
            ("proposed", 0.030, 0.80),
            ("baseline", 0.020, 0.805),
        ):
            rows.append(
                {
                    "method": method,
                    "attack_type": "conv_decoder",
                    "target_seed": str(seed),
                    "accuracy": str(accuracy),
                    "mse": str(mse),
                }
            )
    rules = {
        "utility_drop_limit": 0.01,
        "bootstrap_iterations": 200,
        "confidence_level": 0.95,
        "minimum_target_pairs": 5,
        "primary_privacy_metrics": ["mse"],
        "privacy_benefit_direction": {"mse": 1},
        "minimum_practical_effect": {"mse": 0.001},
    }
    result = paired_analysis(rows, "proposed", "baseline", rules)["conv_decoder"]
    assert result["paired_target_runs"] == 5
    assert result["utility_ok"]
    assert result["h1_status"] == "SUPPORTED"


def test_secondary_metric_cannot_support_the_primary_hypothesis():
    rows = []
    for seed in range(5):
        for method, mse, lpips in (
            ("proposed", 0.020, 0.50),
            ("baseline", 0.020, 0.20),
        ):
            rows.append(
                {
                    "method": method,
                    "attack_type": "conv_decoder",
                    "target_seed": str(seed),
                    "accuracy": "0.8",
                    "mse": str(mse),
                    "lpips": str(lpips),
                }
            )
    rules = {
        "utility_drop_limit": 0.01,
        "bootstrap_iterations": 200,
        "confidence_level": 0.975,
        "minimum_target_pairs": 5,
        "primary_privacy_metrics": ["mse"],
        "privacy_benefit_direction": {"mse": 1, "lpips": 1},
        "minimum_practical_effect": {"mse": 0.001, "lpips": 0.01},
    }
    result = paired_analysis(rows, "proposed", "baseline", rules)["conv_decoder"]
    assert result["supported_privacy_metrics"] == ["lpips"]
    assert result["supported_primary_privacy_metrics"] == []
    assert result["h1_status"] == "NOT_SUPPORTED"
