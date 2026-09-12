from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def scheduled_privacy_weight(
    base_weight: float,
    epoch: int,
    *,
    warmup_epochs: int = 0,
    ramp_epochs: int = 0,
    warmup_factor: float = 0.0,
) -> float:
    """Return the privacy weight for a one-based training epoch."""
    if base_weight < 0:
        raise ValueError("base_weight must be non-negative")
    if epoch <= 0:
        raise ValueError("epoch must be one-based and positive")
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("warmup_epochs and ramp_epochs must be non-negative")
    if not 0.0 <= warmup_factor <= 1.0:
        raise ValueError("warmup_factor must be in [0, 1]")
    if epoch <= warmup_epochs:
        return base_weight * warmup_factor
    if ramp_epochs == 0:
        return base_weight
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    factor = warmup_factor + (1.0 - warmup_factor) * progress
    return base_weight * factor


@dataclass(frozen=True)
class GradientComposition:
    gradient: Tensor
    conflict_fraction: Tensor
    mean_cosine: Tensor


def task_priority_gradient(
    task_gradient: Tensor,
    privacy_gradient: Tensor,
    privacy_weight: float,
    numerical_epsilon: float = 1e-12,
) -> GradientComposition:
    """Remove privacy-gradient components that oppose the task gradient.

    Projection is performed per sample at the smashed representation. The task
    gradient is left unchanged, while non-conflicting privacy components retain
    their configured weight.
    """
    if task_gradient.shape != privacy_gradient.shape:
        raise ValueError("task and privacy gradients must have identical shapes")
    if task_gradient.ndim < 2:
        raise ValueError("gradients must include batch and feature dimensions")
    if privacy_weight < 0:
        raise ValueError("privacy_weight must be non-negative")

    task_flat = task_gradient.flatten(start_dim=1)
    privacy_flat = privacy_gradient.flatten(start_dim=1)
    dot = (task_flat * privacy_flat).sum(dim=1, keepdim=True)
    task_norm_sq = task_flat.square().sum(dim=1, keepdim=True)
    privacy_norm = privacy_flat.norm(dim=1, keepdim=True)
    cosine = dot / (
        task_norm_sq.sqrt() * privacy_norm
    ).clamp_min(numerical_epsilon)

    conflicting = dot < 0
    coefficient = torch.where(
        conflicting,
        dot / task_norm_sq.clamp_min(numerical_epsilon),
        torch.zeros_like(dot),
    )
    adjusted_privacy = privacy_flat - coefficient * task_flat
    composed = task_flat + privacy_weight * adjusted_privacy
    return GradientComposition(
        gradient=composed.reshape_as(task_gradient),
        conflict_fraction=conflicting.to(task_gradient.dtype).mean(),
        mean_cosine=cosine.mean(),
    )
