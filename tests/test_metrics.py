import torch

from publication_cem.metrics import reconstruction_metrics


def test_identity_reconstruction_metrics() -> None:
    image = torch.rand(2, 3, 16, 16)
    metrics = reconstruction_metrics(image, image)
    assert metrics["mse"].item() == 0.0
    assert metrics["ssim"].item() > 0.999
    assert metrics["psnr"].item() > 100
