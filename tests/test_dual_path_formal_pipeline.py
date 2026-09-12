import json

import torch
from PIL import Image

import scripts.run_dual_path_publication_pipeline as formal_pipeline
from scripts.aggregate_dual_path_publication import (
    bootstrap_mean_ci,
    build_summary,
    write_narratives,
)
from scripts.evaluate_dual_path_utility import aggregate_runs
from scripts.export_dual_path_qualitative_grid import render


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_utility_gate_uses_worst_repeated_accuracy() -> None:
    result = aggregate_runs(
        [
            {"fused_accuracy": 0.82},
            {"fused_accuracy": 0.81},
            {"fused_accuracy": 0.83},
        ],
        0.8033,
    )
    assert result["status"] == "PASS"
    assert result["conservative_repeated_validation_accuracy"] == 0.81


def test_bootstrap_interval_is_reproducible() -> None:
    first = bootstrap_mean_ci([0.81, 0.82, 0.83], draws=1_000)
    second = bootstrap_mean_ci([0.81, 0.82, 0.83], draws=1_000)
    assert first == second
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_qualitative_grid_defaults_to_publication_resolution(tmp_path) -> None:
    images = torch.rand(2, 3, 64, 64)
    output = tmp_path / "grid.png"
    render([("Original", images), ("Decoder", images)], output)
    with Image.open(output) as grid:
        assert grid.width >= 1000
        assert grid.height >= 500


def test_stage2_command_has_one_output_directory(monkeypatch, tmp_path) -> None:
    commands = {}

    def capture_job(
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

    monkeypatch.setattr(formal_pipeline, "run_job", capture_job)
    formal_pipeline.train_target(
        {},
        tmp_path / "state.json",
        tmp_path / "data",
        tmp_path / "legacy",
        127,
        3,
        0.0,
    )

    command = commands["target_seed127_stage2"]
    output_dir = str(formal_pipeline.RESULTS_ROOT / "targets" / "seed127" / "stage2")
    assert command[command.index("--output-dir") + 1] == output_dir
    assert command.count(output_dir) == 1


def test_formal_summary_requires_all_published_metrics(tmp_path) -> None:
    results = tmp_path / "results"
    target_seed = 125
    attacker_seed = 10125
    write_json(
        results / "targets" / "seed125" / "utility_l031_s010.json",
        {
            "mean_repeated_validation_accuracy": 0.82,
            "conservative_repeated_validation_accuracy": 0.81,
            "utility_runs": [
                {
                    "legacy_accuracy": 0.70,
                    "semantic_accuracy": 0.75,
                    "fused_accuracy": 0.82,
                }
            ],
        },
    )
    write_json(
        results / "targets" / "seed125" / "utility_stage1_l031_s010.json",
        {
            "mean_repeated_validation_accuracy": 0.80,
            "conservative_repeated_validation_accuracy": 0.79,
        },
    )
    write_json(
        results / "targets" / "seed125" / "efficiency_profile.json",
        {
            "client_parameters": 1,
            "server_parameters": 1,
            "total_payload_elements": 1,
            "transmitted_bytes_per_sample_fp32": 4,
            "client_latency_ms_per_sample_mean": 1.0,
            "server_latency_ms_per_sample_mean": 1.0,
            "end_to_end_latency_ms_per_sample_mean": 2.0,
        },
    )
    for attack in ("decoder", "gan", "adaptive"):
        for knowledge in ("training", "inference"):
            write_json(
                results
                / "attacks"
                / "target_seed125"
                / "attacker_seed10125"
                / f"{attack}_{knowledge}"
                / "attack_metrics.json",
                {
                    "evaluation": {
                        "mse": 0.04,
                        "mae": 0.1,
                        "psnr": 14.0,
                        "ssim": 0.5,
                        "lpips": 0.4,
                        "identity_cosine_similarity": 0.3,
                    }
                },
            )
    summary = build_summary(results, [target_seed], [attacker_seed])
    assert summary["status"] == "PASS"
    assert summary["strictly_dominates_published_mean"] is True
    assert all(summary["checks"].values())

    paper_output = tmp_path / "paper"
    write_narratives(summary, paper_output)
    main_narrative = (paper_output / "main_results_narrative.tex").read_text()
    assert "All five means were above the corresponding published reference" in main_narrative
    assert (paper_output / "attack_results_narrative.tex").is_file()
    assert (paper_output / "ablation_results_narrative.tex").is_file()
    assert (paper_output / "efficiency_results_narrative.tex").is_file()
    assert "Each mean exceeded the corresponding published" in (
        paper_output / "abstract_results.tex"
    ).read_text()
    assert (paper_output / "conclusion_results.tex").is_file()

    failed_summary = json.loads(json.dumps(summary))
    failed_summary["checks"]["decoder_inference_mse"] = False
    failed_summary["confidence_checks"]["decoder_inference_mse"] = False
    failed_output = tmp_path / "failed-paper"
    write_narratives(failed_summary, failed_output)
    failed_narrative = (failed_output / "main_results_narrative.tex").read_text()
    assert "decoder inference knowledge" in failed_narrative
    assert "decoder_inference_mse" not in failed_narrative
    assert "At least one pre-specified accuracy or MSE reference was not exceeded" in (
        failed_output / "abstract_results.tex"
    ).read_text()
