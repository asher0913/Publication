from scripts.check_dual_path_pareto_gate import evaluate_gate


BASELINE = {
    "accuracy": 0.8033,
    "decoder_training_mse": 0.0182,
    "decoder_inference_mse": 0.0211,
    "gan_training_mse": 0.0212,
    "gan_inference_mse": 0.0231,
}


def test_gate_requires_all_five_metrics_to_improve() -> None:
    result = evaluate_gate(
        {"best_validation_accuracy": 0.81},
        {"training": {"mse": 0.02}, "inference": {"mse": 0.022}},
        {"training": {"mse": 0.022}, "inference": {"mse": 0.024}},
        BASELINE,
    )
    assert result["status"] == "PASS"


def test_gate_rejects_utility_only_improvement() -> None:
    result = evaluate_gate(
        {"best_validation_accuracy": 0.85},
        {"training": {"mse": 0.01}, "inference": {"mse": 0.01}},
        {"training": {"mse": 0.01}, "inference": {"mse": 0.01}},
        BASELINE,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["accuracy"] is True
    assert result["checks"]["decoder_inference_mse"] is False
