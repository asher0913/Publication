from scripts.validate_formal_environment import EXPECTED, validate_versions


def test_formal_environment_validator_rejects_version_or_cuda_drift():
    observed = {
        **EXPECTED,
        "cuda_available": True,
        "gpus": ["test-gpu"],
        "cudnn": 8900,
    }
    assert validate_versions(observed) == []
    observed["torch"] = "2.5.1"
    observed["cuda_available"] = False
    mismatches = validate_versions(observed)
    assert any("torch" in message for message in mismatches)
    assert any("cuda_available" in message for message in mismatches)
