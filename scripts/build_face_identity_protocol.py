#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_protocol_loaders
from publication_cem.evaluators import FaceEmbedder
from publication_cem.reproducibility import (
    module_state_sha256,
    save_json,
    seed_everything,
)


def resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def class_prototypes(embedder, loader, num_classes: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    totals = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for images, labels in loader:
        embeddings = embedder(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        if totals is None:
            totals = torch.zeros(
                num_classes,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=device,
            )
        totals.index_add_(0, labels, embeddings)
        counts.index_add_(
            0,
            labels,
            torch.ones(labels.shape, dtype=counts.dtype, device=device),
        )
    if totals is None or torch.any(counts == 0):
        raise ValueError("every identity needs at least one target-train image")
    prototypes = F.normalize(totals / counts[:, None], p=2, dim=1)
    return prototypes, counts.to(dtype=torch.long)


@torch.no_grad()
def calibrate_threshold(
    embedder,
    loader,
    prototypes: torch.Tensor,
    target_far: float,
    device,
) -> dict:
    impostor_scores = []
    genuine_scores = []
    top1_correct = 0
    count = 0
    for images, labels in loader:
        embeddings = embedder(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        scores = embeddings @ prototypes.T
        genuine_scores.append(scores.gather(1, labels[:, None]).squeeze(1).cpu())
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, labels[:, None], False)
        impostor_scores.append(scores[mask].cpu())
        top1_correct += int(scores.argmax(dim=1).eq(labels).sum())
        count += labels.numel()
    if not count:
        raise ValueError("target-validation loader is empty")
    impostor = torch.cat(impostor_scores)
    genuine = torch.cat(genuine_scores)
    rank = max(1, math.ceil((1.0 - target_far) * impostor.numel()))
    threshold = float(impostor.kthvalue(rank).values)
    return {
        "verification_threshold": threshold,
        "validation_empirical_far": float((impostor >= threshold).float().mean()),
        "validation_tar_at_far": float((genuine >= threshold).float().mean()),
        "validation_identity_top1": top1_correct / count,
        "validation_images": count,
        "validation_impostor_comparisons": impostor.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/facescrub_base.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/face_identity_protocol.pt")
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if not 0 < args.target_far < 1:
        raise ValueError("target FAR must be in (0, 1)")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    split_manifest = resolve_from_root(data_cfg["split_manifest"])
    if not split_manifest.is_file():
        raise FileNotFoundError("a frozen split manifest is required")
    seed_everything(int(config["seed"]))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_data_cfg = dict(data_cfg)
    resolved_data_cfg["root"] = str(resolve_from_root(data_cfg["root"]))
    resolved_data_cfg["split_manifest"] = str(split_manifest)
    if args.batch_size is not None:
        resolved_data_cfg["batch_size"] = args.batch_size
    if args.workers is not None:
        resolved_data_cfg["workers"] = args.workers
    loaders = build_protocol_loaders(resolved_data_cfg, int(config["seed"]))
    embedder = FaceEmbedder(device)
    prototypes, counts = class_prototypes(
        embedder, loaders.statistics, loaders.num_classes, device
    )
    calibration = calibrate_threshold(
        embedder,
        loaders.validation,
        prototypes,
        args.target_far,
        device,
    )
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "embedding_model": "facenet-pytorch/InceptionResnetV1-vggface2",
        "embedding_model_weights_sha256": module_state_sha256(embedder.model),
        "input_size": embedder.input_size,
        "input_standardisation": "(x_[0,1] * 255 - 127.5) / 128",
        "resize_antialias": device.type != "mps",
        "calibration_device": str(device),
        "formal_cuda_compatible": device.type == "cuda",
        "class_to_index": loaders.class_to_index,
        "prototypes": prototypes.cpu(),
        "prototype_counts": counts.cpu(),
        "target_far": args.target_far,
        "verification_threshold": calibration["verification_threshold"],
        "split_manifest_sha256": file_sha256(split_manifest),
        "target_train_path_sha256": manifest["splits"]["target_train"]["sha256"],
        "target_validation_path_sha256": manifest["splits"]["target_validation"]["sha256"],
        "calibration": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"prototypes", "prototype_counts", "class_to_index"}
    }
    try:
        protocol_reference = str(args.output.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        protocol_reference = str(args.output.resolve())
    metadata.update(
        protocol_file=protocol_reference,
        protocol_file_sha256=file_sha256(args.output),
        num_classes=loaders.num_classes,
        prototype_count_min=int(counts.min()),
        prototype_count_max=int(counts.max()),
        status="CALIBRATED",
    )
    save_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
