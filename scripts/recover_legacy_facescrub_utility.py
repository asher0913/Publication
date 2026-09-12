#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

# The archived VGG module imports THOP for an unrelated profiling method. Keep
# recovery runnable in the frozen experiment environment without adding a
# training-time dependency that is never used here.
try:
    import thop  # noqa: F401
except ModuleNotFoundError:
    thop_stub = types.ModuleType("thop")

    def unavailable_profile(*_args, **_kwargs):
        raise RuntimeError("THOP is required only when legacy FLOP profiling is used")

    thop_stub.profile = unavailable_profile
    sys.modules["thop"] = thop_stub

from model_architectures.vgg import vgg11_bn_sgm
from publication_cem.server_adapter import ResidualSemanticAdapter


FACESCRUB_MEAN = (0.5708, 0.5905, 0.4272)
FACESCRUB_STD = (0.2058, 0.2275, 0.2098)
PUBLISHED_ACCURACY = 0.8033


class SilentLogger:
    def debug(self, *_args, **_kwargs) -> None:
        return None

    info = debug
    warning = debug


class ServerRecoveryModel(nn.Module):
    def __init__(
        self,
        adapter: nn.Module,
        cloud: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.cloud = cloud
        self.classifier = classifier

    def forward(self, smashed_features: Tensor) -> Tensor:
        hidden = self.cloud(self.adapter(smashed_features))
        if hidden.ndim == 4:
            hidden = F.adaptive_avg_pool2d(hidden, output_size=1)
        return self.classifier(hidden.flatten(start_dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--adapter-hidden-channels", type=int, default=64)
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


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_loaders(data_root: Path, batch_size: int, workers: int) -> tuple:
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(FACESCRUB_MEAN, FACESCRUB_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(FACESCRUB_MEAN, FACESCRUB_STD),
        ]
    )
    train = datasets.ImageFolder(data_root / "train", transform=train_transform)
    validation = datasets.ImageFolder(data_root / "val", transform=validation_transform)
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("FaceScrub train/validation class mappings differ")
    generator = torch.Generator().manual_seed(125)
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


def load_legacy_model(checkpoint_dir: Path, device: torch.device) -> nn.Module:
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
        torch.load(
            checkpoint_dir / "checkpoint_classifier_best.tar", map_location="cpu"
        )
    )
    return model.to(device)


@torch.no_grad()
def evaluate(
    local: nn.Module,
    server: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    seed: int,
) -> dict:
    local.eval()
    server.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    correct = 0
    count = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smashed = local(images)
        noise = torch.randn(
            smashed.shape,
            device=device,
            dtype=smashed.dtype,
            generator=generator,
        )
        logits = server(smashed + noise_std * noise)
        loss_sum += float(F.cross_entropy(logits, labels)) * images.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += images.shape[0]
    return {"accuracy": correct / count, "loss": loss_sum / count, "count": count}


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.noise_std < 0:
        raise ValueError("epochs must be positive and noise_std non-negative")
    if not 0 <= args.distillation_weight <= 1:
        raise ValueError("distillation weight must be in [0, 1]")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to append to existing run: {log_path}")

    train_loader, validation_loader, class_to_index = build_loaders(
        args.data_root, args.batch_size, args.workers
    )
    legacy = load_legacy_model(args.checkpoint_dir, device)
    local = legacy.local.eval().requires_grad_(False)
    local_hash = state_sha256(local)

    teacher = ServerRecoveryModel(
        nn.Identity(),
        copy.deepcopy(legacy.cloud),
        copy.deepcopy(legacy.classifier),
    ).to(device)
    teacher.eval().requires_grad_(False)
    student = ServerRecoveryModel(
        ResidualSemanticAdapter(16, args.adapter_hidden_channels),
        legacy.cloud,
        legacy.classifier,
    ).to(device)

    baseline = evaluate(
        local,
        teacher,
        validation_loader,
        device,
        args.noise_std,
        args.seed + 100_000,
    )
    save_json(args.output_dir / "baseline_metrics.json", baseline)

    adapter_parameters = list(student.adapter.parameters())
    cloud_parameters = list(student.cloud.parameters())
    classifier_parameters = list(student.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": args.learning_rate},
            {"params": cloud_parameters, "lr": args.learning_rate * 0.25},
            {"params": classifier_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    best_accuracy = baseline["accuracy"]
    best_epoch = 0
    torch.save(
        {
            "epoch": 0,
            "server_state": student.state_dict(),
            "local_sha256": local_hash,
            "validation": baseline,
            "class_to_index": class_to_index,
            "args": vars(args),
        },
        args.output_dir / "checkpoint_best.pt",
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                smashed = local(images)
                teacher_logits = teacher(smashed)
            noisy_smashed = smashed + args.noise_std * torch.randn_like(smashed)
            student_logits = student(noisy_smashed)
            ce = F.cross_entropy(student_logits, labels, label_smoothing=0.05)
            kd = distillation_loss(student_logits, teacher_logits, args.temperature)
            loss = (1.0 - args.distillation_weight) * ce + args.distillation_weight * kd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_ce += float(ce.detach()) * batch_size
            total_kd += float(kd.detach()) * batch_size
            total_count += batch_size

        scheduler.step()
        validation = evaluate(
            local,
            student,
            validation_loader,
            device,
            args.noise_std,
            args.seed + 100_000,
        )
        if state_sha256(local) != local_hash:
            raise RuntimeError("frozen local encoder changed during server recovery")
        record = {
            "epoch": epoch,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "train_loss": total_loss / total_count,
            "train_cross_entropy": total_ce / total_count,
            "train_distillation": total_kd / total_count,
            "validation_accuracy": validation["accuracy"],
            "validation_loss": validation["loss"],
            "local_sha256": local_hash,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "server_state": student.state_dict(),
                    "local_sha256": local_hash,
                    "validation": validation,
                    "class_to_index": class_to_index,
                    "args": vars(args),
                },
                args.output_dir / "checkpoint_best.pt",
            )

    result = {
        "status": "PASS" if best_accuracy > args.utility_threshold else "FAIL",
        "baseline_accuracy": baseline["accuracy"],
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "published_accuracy_threshold": args.utility_threshold,
        "strictly_exceeds_published_accuracy": best_accuracy
        > args.utility_threshold,
        "local_encoder_frozen": True,
        "local_sha256": local_hash,
        "privacy_result_carried_forward_only_for_identical_local_encoder": True,
    }
    save_json(args.output_dir / "utility_gate.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
