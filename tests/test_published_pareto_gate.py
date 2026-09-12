import sys

from scripts.check_published_pareto_gate import evaluate_gate, parse_args


BASELINE = {
    "accuracy": 0.85,
    "decoder_inference_mse": 0.0035,
    "gan_inference_mse": 0.0037,
}


def test_gate_requires_every_metric_to_strictly_improve() -> None:
    result = evaluate_gate(0.86, 0.004, 0.0042, BASELINE)
    assert result["status"] == "PASS"
    assert result["strictly_dominates"] is True


def test_gate_fails_when_only_privacy_improves() -> None:
    result = evaluate_gate(0.84, 0.02, 0.03, BASELINE)
    assert result["status"] == "FAIL"
    assert result["checks"] == {
        "accuracy": False,
        "decoder_inference_mse": True,
        "gan_inference_mse": True,
    }


def test_cli_defaults_to_the_published_noise_arl_cem_baseline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_published_pareto_gate.py",
            "--target-summary",
            "target.json",
            "--decoder-metrics",
            "decoder.json",
            "--gan-metrics",
            "gan.json",
            "--output",
            "gate.json",
        ],
    )
    assert parse_args().baseline == "noise_arl_cem"
