from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .reproducibility import module_state_sha256


class LPIPSMetric:
    """Frozen LPIPS evaluator returning one distance per image."""

    def __init__(
        self,
        device: torch.device | str,
        net: str = "alex",
        version: str = "0.1",
        model: nn.Module | None = None,
    ) -> None:
        if model is None:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError(
                    "LPIPS evaluation requires `pip install -e '.[evaluation]'`"
                ) from exc
            model = lpips.LPIPS(net=net, version=version, verbose=False)
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("LPIPS prediction and target shapes must match")
        prediction = prediction.clamp(0, 1).mul(2).sub(1)
        target = target.clamp(0, 1).mul(2).sub(1)
        distances = self.model(prediction, target)
        return distances.reshape(distances.shape[0], -1).mean(dim=1)


class FaceEmbedder:
    """Frozen VGGFace2 FaceNet embedding with explicit input standardisation."""

    def __init__(
        self,
        device: torch.device | str,
        model: nn.Module | None = None,
        input_size: int = 160,
    ) -> None:
        if model is None:
            try:
                from facenet_pytorch import InceptionResnetV1
            except ImportError as exc:
                raise RuntimeError(
                    "Face evaluation requires `pip install -e '.[evaluation]'`"
                ) from exc
            model = InceptionResnetV1(pretrained="vggface2", classify=False)
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.input_size = input_size

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images.clamp(0, 1),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=images.device.type != "mps",
        )
        # facenet-pytorch's fixed_image_standardization for [0, 255] images.
        standardised = (images.mul(255.0) - 127.5) / 128.0
        embeddings = self.model(standardised)
        if embeddings.ndim != 2:
            embeddings = embeddings.flatten(start_dim=1)
        return F.normalize(embeddings, p=2, dim=1)


@dataclass(frozen=True)
class FaceIdentityBatch:
    top1_success: Tensor
    original_reconstruction_cosine: Tensor
    true_prototype_similarity: Tensor
    verification_accept: Tensor


class FaceIdentityEvaluator:
    """Evaluate reconstructed identities against a frozen, pre-calibrated protocol."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        prototypes: Tensor,
        verification_threshold: float,
    ) -> None:
        if prototypes.ndim != 2:
            raise ValueError("face prototypes must have shape [classes, embedding_dim]")
        self.embedder = embedder
        self.prototypes = F.normalize(prototypes, p=2, dim=1)
        self.verification_threshold = float(verification_threshold)

    @classmethod
    def from_protocol(
        cls,
        path: Path,
        device: torch.device | str,
        expected_class_to_index: dict[str, int],
        expected_target_far: float,
        embedder: FaceEmbedder | None = None,
    ) -> "FaceIdentityEvaluator":
        payload: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=False
        )
        if payload.get("class_to_index") != expected_class_to_index:
            raise ValueError("face protocol class mapping differs from target checkpoint")
        if payload.get("embedding_model") != "facenet-pytorch/InceptionResnetV1-vggface2":
            raise ValueError("unsupported face identity protocol model")
        if float(payload.get("target_far", -1)) != float(expected_target_far):
            raise ValueError("face protocol target FAR differs from attack configuration")
        resolved_embedder = embedder or FaceEmbedder(device)
        if module_state_sha256(resolved_embedder.model) != payload.get(
            "embedding_model_weights_sha256"
        ):
            raise ValueError("loaded face model weights differ from calibration protocol")
        return cls(
            embedder=resolved_embedder,
            prototypes=payload["prototypes"].to(device),
            verification_threshold=float(payload["verification_threshold"]),
        )

    @torch.no_grad()
    def __call__(
        self,
        reconstruction: Tensor,
        original: Tensor,
        labels: Tensor,
    ) -> FaceIdentityBatch:
        reconstructed_embeddings = self.embedder(reconstruction)
        original_embeddings = self.embedder(original)
        prototypes = self.prototypes.to(reconstructed_embeddings.device)
        similarities = reconstructed_embeddings @ prototypes.T
        labels = labels.to(similarities.device, dtype=torch.long)
        true_similarity = similarities.gather(1, labels[:, None]).squeeze(1)
        return FaceIdentityBatch(
            top1_success=similarities.argmax(dim=1).eq(labels).float(),
            original_reconstruction_cosine=(
                reconstructed_embeddings * original_embeddings
            ).sum(dim=1),
            true_prototype_similarity=true_similarity,
            verification_accept=(
                true_similarity >= self.verification_threshold
            ).float(),
        )
