import torch
from torch import nn

from publication_cem.evaluators import FaceEmbedder, FaceIdentityEvaluator, LPIPSMetric


class FakeLPIPS(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)


class FakeFaceModel(nn.Module):
    def forward(self, images):
        return torch.stack(
            (
                images[:, 0].mean(dim=(1, 2)),
                images[:, 1].mean(dim=(1, 2)),
            ),
            dim=1,
        )


def test_lpips_normalises_range_and_returns_per_image_values():
    metric = LPIPSMetric("cpu", model=FakeLPIPS())
    prediction = torch.zeros(2, 3, 8, 8)
    target = torch.ones_like(prediction)
    scores = metric(prediction, target)
    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.full((2,), 4.0))


def test_face_identity_metrics_use_frozen_prototypes_and_threshold():
    embedder = FaceEmbedder("cpu", model=FakeFaceModel(), input_size=8)
    evaluator = FaceIdentityEvaluator(
        embedder,
        prototypes=torch.eye(2),
        verification_threshold=0.8,
    )
    originals = torch.full((2, 3, 8, 8), 0.5)
    originals[0, 0] = 1
    originals[1, 1] = 1
    reconstructions = originals.clone()
    result = evaluator(reconstructions, originals, torch.tensor([0, 1]))
    assert torch.equal(result.top1_success, torch.ones(2))
    assert torch.allclose(result.original_reconstruction_cosine, torch.ones(2))
    assert torch.allclose(result.true_prototype_similarity, torch.ones(2))
    assert torch.equal(result.verification_accept, torch.ones(2))
