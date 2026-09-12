from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PrototypeCEMConfig:
    """Configuration for the publication-candidate CEM estimator.

    The projection is fixed and is used only for prototype assignment. The
    variance entering the privacy objective is always measured in the full
    smashed-feature space.
    """

    feature_dim: int
    num_classes: int
    num_slots: int = 8
    projection_dim: int = 64
    noise_std: float = 0.025
    temperature: float = 0.10
    prototype_momentum: float = 0.95
    loss_mode: str = "mi_bound"
    variance_threshold: float = 0.15
    threshold_offset: float = 0.01
    numerical_epsilon: float = 1e-8
    detach_assignments: bool = True
    update_prototypes: bool = True
    min_prototypes_for_loss: int = 1
    projector_mode: str = "fixed"
    variance_space: str = "full"
    seed: int = 125

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if not 0 < self.projection_dim <= self.feature_dim:
            raise ValueError("projection_dim must be in [1, feature_dim]")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive for CEM")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.prototype_momentum < 1:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if self.loss_mode not in {"mi_bound", "threshold"}:
            raise ValueError("loss_mode must be 'mi_bound' or 'threshold'")
        if self.variance_threshold <= 0:
            raise ValueError("variance_threshold must be positive")
        if self.threshold_offset <= 0 or self.numerical_epsilon <= 0:
            raise ValueError("stability constants must be positive")
        if not 1 <= self.min_prototypes_for_loss <= self.num_slots:
            raise ValueError("min_prototypes_for_loss must be in [1, num_slots]")
        if self.projector_mode not in {"fixed", "learned"}:
            raise ValueError("projector_mode must be 'fixed' or 'learned'")
        if self.variance_space not in {"full", "projected"}:
            raise ValueError("variance_space must be 'full' or 'projected'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
