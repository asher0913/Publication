from scripts.run_supervisor_revision_experiments import point_name


def test_noise_point_name_is_stable_and_path_safe() -> None:
    assert point_name(0.31, 0.10) == "g031_s010"
