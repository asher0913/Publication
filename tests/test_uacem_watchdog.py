import hashlib
import json

import pytest

from scripts import run_uacem_watchdog as watchdog


def test_evidence_ready_requires_audit_bundle_manifest_and_matching_hash(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    (tmp_path / "results/uacem_formal").mkdir(parents=True)
    (tmp_path / "handoff").mkdir()
    (tmp_path / "results/uacem_formal/audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    bundle = tmp_path / "handoff/uacem_paper_writing_bundle.tar.gz"
    bundle.write_bytes(b"evidence")
    digest = hashlib.sha256(b"evidence").hexdigest()
    (tmp_path / "handoff/uacem_paper_writing_bundle.json").write_text(
        json.dumps({"status": "EXPERIMENT_EVIDENCE_READY", "sha256": digest}),
        encoding="utf-8",
    )
    ready, checks = watchdog.evidence_ready()
    assert ready
    assert all(checks.values())


def test_singleton_lock_rejects_second_watchdog(tmp_path):
    first = watchdog.acquire_singleton(tmp_path / "watchdog.lock")
    try:
        with pytest.raises(RuntimeError, match="already holds the lock"):
            watchdog.acquire_singleton(tmp_path / "watchdog.lock")
    finally:
        first.close()
