import argparse
import sys

import pytest

from scripts.run_full_publication_pipeline import (
    build_pipeline_steps,
    parse_gpu_indices,
    step_fingerprint,
)


def test_full_pipeline_is_one_ordered_campaign_with_complete_handoff():
    steps = build_pipeline_steps((0, 1))
    names = [step.name for step in steps]
    assert len(steps) == 23
    assert names[:3] == ["01_preflight", "02_generate_pilot", "03_pilot_targets"]
    assert names[-3:] == [
        "21_aggregate_and_generate_assets",
        "22_final_evidence_audit",
        "23_build_paper_handoff",
    ]
    attack_step = next(step for step in steps if step.name == "13_headline_attacks")
    assert len(attack_step.processes) == 2
    assert all("--shard-count" in process.command for process in attack_step.processes)
    assert step_fingerprint(attack_step) == step_fingerprint(attack_step)
    assert attack_step.processes[0].command[0] == sys.executable


def test_gpu_index_parser_rejects_duplicates_and_invalid_values():
    assert parse_gpu_indices("0, 2") == (0, 2)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpu_indices("0,0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpu_indices("gpu0")
