import json
from pathlib import Path

from scripts.generate_uacem_formal_runbooks import (
    ablation_jobs,
    generalisation_jobs,
    main_jobs,
    paper_output_jobs,
)
from scripts.run_uacem_publication_pipeline import build_uacem_steps


ROOT = Path(__file__).resolve().parents[1]


def test_uacem_formal_protocol_expands_to_frozen_evidence_campaign(tmp_path):
    protocol = json.loads(
        (ROOT / "configs/uacem_formal_protocol.json").read_text(encoding="utf-8")
    )
    stages = {}
    stages.update(main_jobs(protocol, tmp_path))
    stages.update(generalisation_jobs(protocol, tmp_path))
    stages.update(ablation_jobs(protocol))
    stages.update(paper_output_jobs())
    assert len(stages["main_targets"]) == 15
    assert len(stages["main_attacks"]) == 150
    assert len(stages["generalisation_attacks"]) == 72
    assert len(stages["ablation_attacks"]) == 15
    assert len(stages["paper_outputs"]) == 9
    assert sum(len(jobs) for jobs in stages.values()) == 419
    assert protocol["selection"]["target_test_accessed"] is False
    assert protocol["selection"]["alpha"] == 0.3
    assert protocol["selection"]["score"] == "reconstruction_to_task"


def test_uacem_pipeline_orders_calibration_before_formal_attacks():
    steps = build_uacem_steps((0,))
    names = [step.name for step in steps]
    assert names.index("05_main_calibrations") < names.index("07_main_attacks")
    assert names[-1] == "18_paper_outputs"
    assert len(steps) == 18
