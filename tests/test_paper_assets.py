import json

from scripts.generate_paper_assets import summarise, summarise_efficiency


def test_paper_summary_keeps_attack_architectures_separate():
    rows = [
        {
            "method": "candidate",
            "attack_type": attack_type,
            "accuracy": "0.8",
            "mse": mse,
            "ssim": "0.3",
            "psnr": "10",
            "lpips": "0.4",
            "identity_top1_success": "0.2",
        }
        for attack_type, mse in (("conv_decoder", "0.02"), ("gan", "0.03"))
    ]
    summary = summarise(rows)
    assert len(summary) == 2
    assert {row["attack_type"] for row in summary} == {"conv_decoder", "gan"}


def test_efficiency_summary_uses_each_target_once(tmp_path):
    run_dir = tmp_path / "seed125_candidate"
    run_dir.mkdir()
    (run_dir / "training.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "total_epoch_seconds": seconds,
                    "statistics_refresh_seconds": 1.0,
                    "peak_gpu_memory_bytes": 1024**3,
                }
            )
            for seconds in (10.0, 14.0)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "efficiency_profile.json").write_text(
        json.dumps(
            {
                "edge_latency_ms_per_sample_mean": 2.0,
                "transmitted_bytes_per_sample_fp32": 4096,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"method": "candidate", "run_dir": str(run_dir)},
        {"method": "candidate", "run_dir": str(run_dir)},
    ]
    summary, paths = summarise_efficiency(rows)
    assert len(summary) == 1
    assert summary[0]["target_runs"] == 1
    assert summary[0]["total_epoch_seconds_mean"] == 12.0
    assert len(paths) == 2
