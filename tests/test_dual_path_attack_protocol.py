from PIL import Image
import torch
from torch import nn

from scripts.attack_dual_path_facescrub import (
    build_attack_loaders,
    prepare_attacker_inputs,
)
from scripts.combine_dual_path_attack_metrics import combine


def attack_result(knowledge: str, attack_type: str = "decoder") -> dict:
    return {
        "attack_type": attack_type,
        "attack_knowledge": knowledge,
        "dual_path_checkpoint": "target.pt",
        "effective_legacy_noise_std": 0.22,
        "effective_semantic_noise_std": 0.1,
        "evaluation": {"mse": 0.03},
    }


def test_combines_independent_official_attack_splits() -> None:
    result = combine(attack_result("training"), attack_result("inference"))
    assert result["training"]["mse"] == 0.03
    assert result["inference"]["mse"] == 0.03


def test_rejects_swapped_attack_splits() -> None:
    try:
        combine(attack_result("inference"), attack_result("training"))
    except ValueError as error:
        assert "training knowledge" in str(error)
    else:
        raise AssertionError("swapped attack protocols were accepted")


def test_official_split_loader_sizes(tmp_path) -> None:
    for split, count in (("train", 4), ("val", 10)):
        class_dir = tmp_path / split / "identity"
        class_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), color=index).save(class_dir / f"{index}.png")

    training_source, training_evaluation = build_attack_loaders(
        tmp_path, batch_size=2, workers=0, attack_knowledge="training"
    )
    inference_source, inference_evaluation = build_attack_loaders(
        tmp_path, batch_size=2, workers=0, attack_knowledge="inference"
    )
    assert len(training_source.dataset) == 4
    assert len(training_evaluation.dataset) == 10
    assert len(inference_source.dataset) == 9
    assert len(inference_evaluation.dataset) == 1


def test_predicted_identity_condition_uses_semantic_posterior() -> None:
    semantic = nn.Module()
    semantic.classifier = nn.Linear(3, 4, bias=False)
    token = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    attacker_token, condition, posterior = prepare_attacker_inputs(
        semantic, token, labels, "joint", "predicted"
    )
    assert torch.equal(attacker_token, token)
    assert condition is not None
    assert torch.allclose(condition, posterior)
    assert torch.allclose(condition.sum(dim=1), torch.ones(2))


def test_oracle_identity_condition_is_one_hot_upper_bound() -> None:
    semantic = nn.Module()
    semantic.classifier = nn.Linear(3, 4, bias=False)
    token = torch.randn(2, 3)
    labels = torch.tensor([0, 3])
    attacker_token, condition, _posterior = prepare_attacker_inputs(
        semantic, token, labels, "joint", "oracle"
    )
    assert torch.equal(attacker_token, token)
    assert torch.equal(condition, torch.nn.functional.one_hot(labels, 4).float())


def test_spatial_only_attack_removes_sample_specific_semantic_token() -> None:
    semantic = nn.Module()
    semantic.classifier = nn.Linear(3, 4, bias=False)
    token = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    attacker_token, condition, _posterior = prepare_attacker_inputs(
        semantic, token, labels, "spatial", "uniform"
    )
    assert torch.count_nonzero(attacker_token) == 0
    assert condition is not None
    assert torch.allclose(condition, torch.full((2, 4), 0.25))


def test_smoke_limits_do_not_change_class_mapping(tmp_path) -> None:
    for split, count in (("train", 4), ("val", 10)):
        for identity in ("a", "b"):
            class_dir = tmp_path / split / identity
            class_dir.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (8, 8), color=index).save(class_dir / f"{index}.png")
    source, evaluation = build_attack_loaders(
        tmp_path,
        batch_size=2,
        workers=0,
        attack_knowledge="inference",
        max_auxiliary_samples=3,
        max_evaluation_samples=1,
    )
    assert len(source.dataset) == 3
    assert len(evaluation.dataset) == 1
