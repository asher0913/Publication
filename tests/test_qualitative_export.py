import json

from scripts.export_qualitative_grid import METHODS, representative_runs


def test_representative_grid_selection_uses_seed_nearest_group_mean(tmp_path):
    values = [0.01, 0.02, 0.031, 0.04, 0.05]
    for suffix, _ in METHODS:
        for offset, mse in enumerate(values):
            run_dir = tmp_path / f"seed{125 + offset}_{suffix}"
            for attacker_seed in (10125, 20125, 30125):
                metrics_dir = run_dir / f"attacks/conv_decoder/seed{attacker_seed}"
                metrics_dir.mkdir(parents=True)
                (metrics_dir / "attack_metrics.json").write_text(
                    json.dumps({"mse": mse}), encoding="utf-8"
                )
    selected = representative_runs(tmp_path, 10125)
    assert set(selected) == {suffix for suffix, _ in METHODS}
    assert all(record["run_dir"].name.startswith("seed127_") for record in selected.values())
