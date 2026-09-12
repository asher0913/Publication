import torch

from publication_cem.projection import FixedOrthogonalProjector, LearnedProjector


def test_projection_is_fixed_and_orthogonal() -> None:
    projector = FixedOrthogonalProjector(feature_dim=16, projection_dim=4, seed=7)
    assert list(projector.parameters()) == []
    gram = projector.matrix.T @ projector.matrix
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_projection_is_reproducible_for_a_seed() -> None:
    first = FixedOrthogonalProjector(12, 3, seed=19)
    second = FixedOrthogonalProjector(12, 3, seed=19)
    third = FixedOrthogonalProjector(12, 3, seed=20)
    assert torch.equal(first.matrix, second.matrix)
    assert not torch.equal(first.matrix, third.matrix)


def test_learned_projector_is_initially_orthogonal_but_trainable() -> None:
    projector = LearnedProjector(12, 3, seed=19)
    assert list(projector.parameters()) == [projector.matrix]
    loss = projector(torch.ones(2, 12)).square().mean()
    loss.backward()
    assert projector.matrix.grad is not None
