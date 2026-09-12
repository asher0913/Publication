import torch

from publication_cem.dual_path import CalibratedLogitFusion
from scripts.evaluate_dual_path_fusion_ablation import STRATEGIES, aggregate, fuse_logits


def test_fusion_rules_preserve_shape_and_cover_protocol() -> None:
    spatial = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    semantic = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    fusion = CalibratedLogitFusion(semantic_weight=0.35)
    for strategy in STRATEGIES:
        output = fuse_logits(spatial, semantic, fusion, strategy)
        assert output.shape == spatial.shape


def test_confidence_gate_selects_the_more_confident_branch() -> None:
    spatial = torch.tensor([[5.0, 0.0], [0.1, 0.0]])
    semantic = torch.tensor([[0.2, 0.0], [4.0, 0.0]])
    output = fuse_logits(
        spatial, semantic, CalibratedLogitFusion(), "confidence_gating"
    )
    assert torch.equal(output[0], spatial[0])
    assert torch.equal(output[1], semantic[1])


def test_aggregate_reports_across_target_and_noise_repeats() -> None:
    rows = []
    for strategy in STRATEGIES:
        rows.extend(
            [
                {"strategy": strategy, "accuracy": 0.80},
                {"strategy": strategy, "accuracy": 0.82},
            ]
        )
    summary = aggregate(rows)
    assert summary["spatial_only"]["runs"] == 2
    assert abs(summary["spatial_only"]["mean_accuracy"] - 0.81) < 1e-12
