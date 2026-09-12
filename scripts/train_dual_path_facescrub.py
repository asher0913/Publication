#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "archive" / "current_artefact" / "src"))

try:
    import thop  # noqa: F401
except ModuleNotFoundError:
    thop_stub = types.ModuleType("thop")

    def unavailable_profile(*_args, **_kwargs):
        raise RuntimeError("THOP is required only for legacy FLOP profiling")

    thop_stub.profile = unavailable_profile
    sys.modules["thop"] = thop_stub

from model_architectures.vgg import vgg11_bn_sgm  # noqa: E402
from publication_cem.dual_path import (  # noqa: E402
    CalibratedLogitFusion,
    GlobalSemanticBottleneck,
)
from publication_cem.semantic_backbones import (  # noqa: E402
    SEMANTIC_BACKBONES,
    build_semantic_backbone,
    checkpoint_backbone_name,
)


FACESCRUB_MEAN = (0.5708, 0.5905, 0.4272)
FACESCRUB_STD = (0.2058, 0.2275, 0.2098)
PUBLISHED_ACCURACY = 0.8033


class SilentLogger:
    def debug(self, *_args, **_kwargs) -> None:
        return None

    info = debug
    warning = debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--legacy-noise-std", type=float, default=0.025)
    parser.add_argument("--semantic-noise-std", type=float, default=0.05)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument(
        "--semantic-backbone",
        choices=SEMANTIC_BACKBONES,
        default="mobilenet_v3_large",
    )
    parser.add_argument("--semantic-loss-weight", type=float, default=0.35)
    parser.add_argument("--initial-dual-path-checkpoint", type=Path)
    parser.add_argument("--train-legacy-server", action="store_true")
    parser.add_argument("--legacy-server-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--utility-threshold", type=float, default=PUBLISHED_ACCURACY)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_loaders(data_root: Path, batch_size: int, workers: int, seed: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )
    validation_transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )
    train = datasets.ImageFolder(data_root / "train", transform=train_transform)
    validation = datasets.ImageFolder(data_root / "val", transform=validation_transform)
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("FaceScrub train/validation class mappings differ")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(validation, shuffle=False, **common),
        train.class_to_idx,
    )


def normalise_facescrub(images: Tensor) -> Tensor:
    mean = images.new_tensor(FACESCRUB_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(FACESCRUB_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def load_legacy(checkpoint_dir: Path, device: torch.device) -> nn.Module:
    model = vgg11_bn_sgm(
        4,
        SilentLogger(),
        num_client=1,
        num_class=530,
        adds_bottleneck=True,
        bottleneck_option="noRELU_C16S1",
        SCA=True,
        feature_size=16,
    )
    model.local.load_state_dict(
        torch.load(checkpoint_dir / "checkpoint_f_best.tar", map_location="cpu")
    )
    model.cloud.load_state_dict(
        torch.load(checkpoint_dir / "checkpoint_cloud_best.tar", map_location="cpu")
    )
    model.classifier.load_state_dict(
        torch.load(checkpoint_dir / "checkpoint_classifier_best.tar", map_location="cpu")
    )
    return model.to(device).eval().requires_grad_(False)


def restore_legacy_server(legacy: nn.Module, checkpoint: dict) -> None:
    cloud_state = checkpoint.get("legacy_cloud_state")
    classifier_state = checkpoint.get("legacy_classifier_state")
    if (cloud_state is None) != (classifier_state is None):
        raise ValueError("checkpoint contains an incomplete legacy server state")
    if cloud_state is not None:
        legacy.cloud.load_state_dict(cloud_state)
        legacy.classifier.load_state_dict(classifier_state)


def legacy_logits(
    legacy: nn.Module,
    raw_images: Tensor,
    noise_std: float,
    generator=None,
) -> tuple[Tensor, Tensor]:
    smashed = legacy.local(normalise_facescrub(raw_images))
    if noise_std:
        noise = torch.randn(
            smashed.shape,
            device=smashed.device,
            dtype=smashed.dtype,
            generator=generator,
        )
        transmitted = smashed + noise_std * noise
    else:
        transmitted = smashed
    hidden = legacy.cloud(transmitted)
    hidden = F.adaptive_avg_pool2d(hidden, output_size=1).flatten(start_dim=1)
    return legacy.classifier(hidden), transmitted


@torch.no_grad()
def evaluate(
    legacy: nn.Module,
    semantic: GlobalSemanticBottleneck,
    fusion: CalibratedLogitFusion,
    loader: DataLoader,
    device: torch.device,
    legacy_noise_std: float,
    semantic_noise_std: float,
    seed: int,
) -> dict:
    legacy.eval()
    semantic.eval()
    fusion.eval()
    legacy_generator = torch.Generator(device=device).manual_seed(seed)
    semantic_generator = torch.Generator(device=device).manual_seed(seed + 1)
    counts = {"legacy": 0, "semantic": 0, "fused": 0, "total": 0}
    fused_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        old_logits, _ = legacy_logits(
            legacy, images, legacy_noise_std, legacy_generator
        )
        new_logits, _ = semantic(images, semantic_noise_std, semantic_generator)
        fused_logits = fusion(old_logits, new_logits)
        counts["legacy"] += int((old_logits.argmax(dim=1) == labels).sum())
        counts["semantic"] += int((new_logits.argmax(dim=1) == labels).sum())
        counts["fused"] += int((fused_logits.argmax(dim=1) == labels).sum())
        counts["total"] += images.shape[0]
        fused_loss += float(F.cross_entropy(fused_logits, labels)) * images.shape[0]
    return {
        "legacy_accuracy": counts["legacy"] / counts["total"],
        "semantic_accuracy": counts["semantic"] / counts["total"],
        "fused_accuracy": counts["fused"] / counts["total"],
        "fused_loss": fused_loss / counts["total"],
        "count": counts["total"],
        "semantic_weight": float(fusion.mix_logit.sigmoid()),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.token_dim <= 0:
        raise ValueError("epochs and token dimension must be positive")
    if not 0 <= args.semantic_loss_weight <= 1:
        raise ValueError("semantic loss weight must be in [0, 1]")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to append to existing run: {log_path}")

    train_loader, validation_loader, class_to_index = build_loaders(
        args.data_root, args.batch_size, args.workers, args.seed
    )
    legacy = load_legacy(args.legacy_checkpoint_dir, device)
    legacy_local_hash = module_sha256(legacy.local)

    backbone, backbone_channels = build_semantic_backbone(
        args.semantic_backbone,
        pretrained=args.initial_dual_path_checkpoint is None,
    )
    semantic = GlobalSemanticBottleneck(
        backbone,
        backbone_channels=backbone_channels,
        token_dim=args.token_dim,
        num_classes=530,
        dropout=0.1,
    ).to(device)
    fusion = CalibratedLogitFusion(semantic_weight=0.35).to(device)
    if args.initial_dual_path_checkpoint:
        initial = torch.load(
            args.initial_dual_path_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if int(initial["args"]["token_dim"]) != args.token_dim:
            raise ValueError("initial checkpoint token dimension does not match")
        if checkpoint_backbone_name(initial["args"]) != args.semantic_backbone:
            raise ValueError("initial checkpoint semantic backbone does not match")
        if initial["class_to_index"] != class_to_index:
            raise ValueError("initial checkpoint class mapping does not match")
        semantic.load_state_dict(initial["semantic_state"])
        fusion.load_state_dict(initial["fusion_state"])
        restore_legacy_server(legacy, initial)
    parameter_groups = [
        {"params": semantic.backbone.parameters(), "lr": args.backbone_learning_rate},
        {"params": semantic.project.parameters(), "lr": args.learning_rate},
        {"params": semantic.classifier.parameters(), "lr": args.learning_rate},
        {"params": fusion.parameters(), "lr": args.learning_rate},
    ]
    if args.train_legacy_server:
        legacy.cloud.requires_grad_(True)
        legacy.classifier.requires_grad_(True)
        parameter_groups.extend(
            [
                {
                    "params": legacy.cloud.parameters(),
                    "lr": args.legacy_server_learning_rate,
                },
                {
                    "params": legacy.classifier.parameters(),
                    "lr": args.legacy_server_learning_rate,
                },
            ]
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        semantic.train()
        fusion.train()
        legacy.local.eval()
        if args.train_legacy_server:
            legacy.cloud.train()
            legacy.classifier.train()
        else:
            legacy.eval()
        totals = {"loss": 0.0, "fused": 0.0, "semantic": 0.0, "count": 0}
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.train_legacy_server:
                old_logits, _ = legacy_logits(
                    legacy, images, args.legacy_noise_std
                )
            else:
                with torch.no_grad():
                    old_logits, _ = legacy_logits(
                        legacy, images, args.legacy_noise_std
                    )
            new_logits, _ = semantic(images, args.semantic_noise_std)
            fused_logits = fusion(old_logits, new_logits)
            fused_loss = F.cross_entropy(fused_logits, labels, label_smoothing=0.05)
            semantic_loss = F.cross_entropy(
                new_logits, labels, label_smoothing=0.05
            )
            loss = fused_loss + args.semantic_loss_weight * semantic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(semantic.parameters(), max_norm=5.0)
            if args.train_legacy_server:
                nn.utils.clip_grad_norm_(legacy.cloud.parameters(), max_norm=5.0)
                nn.utils.clip_grad_norm_(legacy.classifier.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = images.shape[0]
            totals["loss"] += float(loss.detach()) * batch_size
            totals["fused"] += float(fused_loss.detach()) * batch_size
            totals["semantic"] += float(semantic_loss.detach()) * batch_size
            totals["count"] += batch_size

        scheduler.step()
        metrics = evaluate(
            legacy,
            semantic,
            fusion,
            validation_loader,
            device,
            args.legacy_noise_std,
            args.semantic_noise_std,
            args.seed + 100_000,
        )
        if module_sha256(legacy.local) != legacy_local_hash:
            raise RuntimeError("frozen legacy client changed during dual-path training")
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["count"],
            "train_fused_loss": totals["fused"] / totals["count"],
            "train_semantic_loss": totals["semantic"] / totals["count"],
            **metrics,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if metrics["fused_accuracy"] > best_accuracy:
            best_accuracy = metrics["fused_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "semantic_state": semantic.state_dict(),
                    "fusion_state": fusion.state_dict(),
                    "legacy_local_sha256": legacy_local_hash,
                    "legacy_cloud_state": legacy.cloud.state_dict(),
                    "legacy_classifier_state": legacy.classifier.state_dict(),
                    "legacy_checkpoint_dir": str(args.legacy_checkpoint_dir),
                    "validation": metrics,
                    "class_to_index": class_to_index,
                    "args": vars(args),
                },
                args.output_dir / "checkpoint_best.pt",
            )

    result = {
        "status": "PASS" if best_accuracy > args.utility_threshold else "FAIL",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "published_accuracy_threshold": args.utility_threshold,
        "strictly_exceeds_published_accuracy": best_accuracy
        > args.utility_threshold,
        "transmitted_legacy_encoder_frozen": True,
        "legacy_server_trained": args.train_legacy_server,
        "legacy_local_sha256": legacy_local_hash,
        "transmitted_semantic_token_elements": args.token_dim,
        "privacy_attacks_required": True,
    }
    save_json(args.output_dir / "utility_gate.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
