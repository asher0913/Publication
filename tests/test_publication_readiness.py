from scripts.audit_publication_readiness import (
    audit_repository,
    category_status,
    core_experiments,
    payload,
)


def test_current_matrix_has_25_controlled_target_runs(publication_root):
    _, experiments = core_experiments(publication_root)
    assert len(experiments) == 25
    assert len({experiment["overrides"]["seed"] for experiment in experiments}) == 5


def test_current_repository_is_honestly_reported_as_incomplete(publication_root):
    checks = audit_repository(publication_root, run_checks=False)
    by_id = {check.identifier: check for check in checks}
    assert by_id["source_components"].status == "PASS"
    assert by_id["target_matrix"].status == "PASS"
    assert by_id["independent_data_splits"].status == "PASS"
    assert by_id["headline_target_results"].status == "FAIL"
    assert payload(publication_root, checks)["overall_status"] == "NOT_READY"
    assert "FAIL" in category_status(checks).values()
