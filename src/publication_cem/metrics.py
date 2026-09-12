from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_metrics(prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
    per_image = reconstruction_metrics_per_image(prediction, target)
    return {name: value.mean() for name, value in per_image.items()}


def reconstruction_metrics_per_image(
    prediction: Tensor, target: Tensor
) -> Dict[str, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    mse_per_image = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    psnr_per_image = -10.0 * torch.log10(mse_per_image.clamp_min(1e-12))
    ssim_per_image = structural_similarity(prediction, target)
    return {"mse": mse_per_image, "psnr": psnr_per_image, "ssim": ssim_per_image}


def structural_similarity(
    prediction: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Differentiable SSIM for images in [0, 1], returned per image."""

    if prediction.ndim != 4:
        raise ValueError("SSIM expects NCHW tensors")
    channels = prediction.shape[1]
    coordinates = torch.arange(
        window_size, device=prediction.device, dtype=prediction.dtype
    )
    coordinates = coordinates - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] @ gaussian[None, :]
    window = window_2d.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu_xy
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(torch.finfo(prediction.dtype).eps)
    return score.flatten(start_dim=1).mean(dim=1)


class MetricAccumulator:
    def __init__(self) -> None:
        self.totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        self.count = 0

    def update(self, metrics: Dict[str, Tensor], batch_size: int) -> None:
        for name in self.totals:
            self.totals[name] += float(metrics[name].detach().cpu()) * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {name: math.nan for name in self.totals}
        return {name: value / self.count for name, value in self.totals.items()}
