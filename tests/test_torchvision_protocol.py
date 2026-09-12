from scripts.build_torchvision_protocol import build_index_manifest


def test_index_manifest_partitions_training_and_keeps_official_test_separate():
    train_labels = [label for label in range(3) for _ in range(20)]
    test_labels = [label for label in range(3) for _ in range(4)]
    manifest = build_index_manifest(
        "cifar100",
        train_labels,
        test_labels,
        ["a", "b", "c"],
        seed=125,
        validation_fraction=0.1,
        auxiliary_fraction=0.2,
    )
    train_split_names = (
        "target_train",
        "target_validation",
        "attacker_auxiliary_train",
        "attacker_auxiliary_validation",
    )
    train_sets = [
        set(manifest["splits"][name]["indices"]) for name in train_split_names
    ]
    assert set().union(*train_sets) == set(range(60))
    assert all(
        not left & right
        for position, left in enumerate(train_sets)
        for right in train_sets[position + 1 :]
    )
    assert manifest["splits"]["target_test"]["indices"] == list(range(12))
    assert manifest["splits"]["target_test"]["source"] == "test"
