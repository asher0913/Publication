import torch
from torch import nn

from publication_cem.dual_path import CalibratedLogitFusion, GlobalSemanticBottleneck


def test_semantic_bottleneck_emits_vector_token() -> None:
    backbone = nn.Conv2d(3, 12, kernel_size=3, padding=1)
    model = GlobalSemanticBottleneck(backbone, 12, 8, 5, dropout=0.0)
    logits, token = model(torch.rand(4, 3, 16, 16))
    assert token.shape == (4, 8)
    assert logits.shape == (4, 5)


def test_semantic_noise_is_reproducible() -> None:
    backbone = nn.Conv2d(3, 12, kernel_size=3, padding=1)
    model = GlobalSemanticBottleneck(backbone, 12, 8, 5, dropout=0.0).eval()
    images = torch.rand(2, 3, 16, 16)
    first = model(images, 0.1, torch.Generator().manual_seed(7))[1]
    second = model(images, 0.1, torch.Generator().manual_seed(7))[1]
    assert torch.equal(first, second)


def test_fusion_is_bounded_and_trainable() -> None:
    fusion = CalibratedLogitFusion(semantic_weight=0.25)
    legacy = torch.zeros(2, 3)
    semantic = torch.ones(2, 3)
    output = fusion(legacy, semantic)
    assert torch.allclose(output, torch.full_like(output, 0.25))
    output.sum().backward()
    assert fusion.mix_logit.grad is not None
