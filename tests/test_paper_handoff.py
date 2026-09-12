import json

import pytest

from scripts.build_paper_handoff import collect_files, verify_experiment_evidence


def test_lightweight_handoff_keeps_metrics_but_excludes_checkpoints(tmp_path):
    (tmp_path / "results/run").mkdir(parents=True)
    (tmp_path / "results/run/metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results/run/checkpoint.pt").write_bytes(b"large weights")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    relative = {path.relative_to(tmp_path).as_posix() for path in collect_files(tmp_path)}
    assert "results/run/metrics.json" in relative
    assert "results/run/checkpoint.pt" not in relative
    assert "README.md" in relative


def test_handoff_requires_all_non_manuscript_categories(tmp_path):
    (tmp_path / "results").mkdir()
    audit = {
        "category_status": {
            "engineering": "PASS",
            "protocol": "PASS",
            "preflight": "PASS",
            "evidence": "PASS",
            "manuscript": "FAIL",
        },
        "checks": [],
    }
    statistics = {"status": "ANALYSED", "global_h1_status": "NOT_SUPPORTED"}
    (tmp_path / "results/readiness_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    (tmp_path / "results/headline_statistical_analysis.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )
    loaded_audit, loaded_statistics = verify_experiment_evidence(tmp_path)
    assert loaded_audit["category_status"]["manuscript"] == "FAIL"
    assert loaded_statistics["global_h1_status"] == "NOT_SUPPORTED"

    audit["category_status"]["evidence"] = "FAIL"
    (tmp_path / "results/readiness_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="experiment evidence is incomplete"):
        verify_experiment_evidence(tmp_path)
