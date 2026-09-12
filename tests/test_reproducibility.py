import torch

from publication_cem.reproducibility import tensor_sha256


def test_tensor_hash_changes_with_shape_or_content():
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(matrix) == tensor_sha256(matrix.clone())
    assert tensor_sha256(matrix) != tensor_sha256(matrix.reshape(3, 2))
    changed = matrix.clone()
    changed[0, 0] = 1
    assert tensor_sha256(matrix) != tensor_sha256(changed)
