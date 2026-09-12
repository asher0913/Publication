#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
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
from publication_cem.dual_path import GlobalSemanticBottleneck  # noqa: E402
from publication_cem.semantic_backbones import (  # noqa: E402
    build_semantic_backbone,
    checkpoint_backbone_name,
)
from publication_cem.dual_path_attack import (  # noqa: E402
    DualPathReconstructor,
    IdentityConditionedDualPathReconstructor,
    PatchDiscriminator,
)
from publication_cem.metrics import (  # noqa: E402
    reconstruction_metrics_per_image,
)


FACESCRUB_MEAN = (0.5708, 0.5905, 0.4272)
FACESCRUB_STD = (0.2058, 0.2275, 0.2098)


class SilentLogger:
    def debug(self, *_args, **_kwargs) -> None:
        return None

    info = debug
    warning = debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-path-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        help="override the machine-specific legacy path stored in the target checkpoint",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attack-type", choices=("decoder", "gan"), required=True)
    parser.add_argument(
        "--attack-view",
        choices=("spatial", "joint"),
        default="joint",
        help="representations disclosed to the reconstruction attacker",
    )
    parser.add_argument(
        "--identity-conditioning",
        choices=("none", "uniform", "predicted", "oracle"),
        default="none",
        help=(
            "class prior supplied to a capacity-matched FiLM reconstructor; "
            "oracle is an upper-bound analysis and not a deployable attack"
        ),
    )
    parser.add_argument(
        "--attack-knowledge",
        choices=("training", "inference"),
        required=True,
        help="match the training-time or inference-time attack split in CEM",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-auxiliary-samples",
        type=int,
        help="optional deterministic subset for smoke testing only",
    )
    parser.add_argument(
        "--max-evaluation-samples",
        type=int,
        help="optional deterministic subset for smoke testing only",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--discriminator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--gan-weight", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--initial-decoder-checkpoint", type=Path)
    parser.add_argument("--legacy-noise-std", type=float)
    parser.add_argument("--semantic-noise-std", type=float)
    parser.add_argument("--seed", type=int, default=10125)
    parser.add_argument(
        "--evaluation-noise-seeds",
        nargs="+",
        type=int,
        help=(
            "fixed channel-noise seeds used for final evaluation; the default "
            "preserves the historical single-seed protocol"
        ),
    )
    parser.add_argument(
        "--perceptual-metrics",
        action="store_true",
        help="also evaluate LPIPS and VGGFace2 embedding similarity",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_noise_std(checkpoint_value: float, override: float | None) -> float:
    resolved = checkpoint_value if override is None else override
    if resolved < 0:
        raise ValueError("noise standard deviation must be non-negative")
    return float(resolved)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def normalise_facescrub(images: Tensor) -> Tensor:
    mean = images.new_tensor(FACESCRUB_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(FACESCRUB_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def build_attack_loaders(
    data_root: Path,
    batch_size: int,
    workers: int,
    attack_knowledge: str,
    max_auxiliary_samples: int | None = None,
    max_evaluation_samples: int | None = None,
):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    validation = datasets.ImageFolder(data_root / "val", transform=transform)
    if attack_knowledge == "training":
        auxiliary = datasets.ImageFolder(data_root / "train", transform=transform)
        evaluation = validation
    elif attack_knowledge == "inference":
        split = len(validation) - len(validation) // 10
        auxiliary = Subset(validation, range(0, split))
        evaluation = Subset(validation, range(split, len(validation)))
    else:
        raise ValueError(f"unsupported attack knowledge: {attack_knowledge}")
    if max_auxiliary_samples is not None:
        if max_auxiliary_samples <= 0:
            raise ValueError("maximum auxiliary samples must be positive")
        auxiliary = Subset(auxiliary, range(min(max_auxiliary_samples, len(auxiliary))))
    if max_evaluation_samples is not None:
        if max_evaluation_samples <= 0:
            raise ValueError("maximum evaluation samples must be positive")
        evaluation = Subset(evaluation, range(min(max_evaluation_samples, len(evaluation))))
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(auxiliary, shuffle=True, **common),
        DataLoader(evaluation, shuffle=False, **common),
    )


def load_feature_extractors(
    checkpoint_path: Path,
    device: torch.device,
    legacy_checkpoint_dir: Path | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    legacy_dir = (
        Path(checkpoint["legacy_checkpoint_dir"])
        if legacy_checkpoint_dir is None
        else legacy_checkpoint_dir
    )
    legacy = vgg11_bn_sgm(
        4,
        SilentLogger(),
        num_client=1,
        num_class=530,
        adds_bottleneck=True,
        bottleneck_option="noRELU_C16S1",
        SCA=True,
        feature_size=16,
    )
    legacy.local.load_state_dict(
        torch.load(legacy_dir / "checkpoint_f_best.tar", map_location="cpu")
    )
    legacy = legacy.local.to(device).eval().requires_grad_(False)

    backbone, backbone_channels = build_semantic_backbone(
        checkpoint_backbone_name(args), pretrained=False
    )
    semantic = GlobalSemanticBottleneck(
        backbone,
        backbone_channels=backbone_channels,
        token_dim=int(args["token_dim"]),
        num_classes=530,
        dropout=0.1,
    )
    semantic.load_state_dict(checkpoint["semantic_state"])
    semantic = semantic.to(device).eval().requires_grad_(False)
    return (
        legacy,
        semantic,
        float(args["legacy_noise_std"]),
        float(args["semantic_noise_std"]),
        int(args["token_dim"]),
        dict(checkpoint["class_to_index"]),
    )


def dataset_class_to_index(loader: DataLoader) -> dict[str, int]:
    dataset = loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not hasattr(dataset, "class_to_idx"):
        raise TypeError("attack dataset does not expose an ImageFolder class mapping")
    return dict(dataset.class_to_idx)


@torch.no_grad()
def transmitted_features(
    legacy: nn.Module,
    semantic: GlobalSemanticBottleneck,
    images: Tensor,
    legacy_noise_std: float,
    semantic_noise_std: float,
    generator=None,
) -> tuple[Tensor, Tensor]:
    spatial = legacy(normalise_facescrub(images))
    if legacy_noise_std:
        spatial = spatial + legacy_noise_std * torch.randn(
            spatial.shape,
            device=spatial.device,
            dtype=spatial.dtype,
            generator=generator,
        )
    token = semantic.encode(images)
    if semantic_noise_std:
        token = token + semantic_noise_std * torch.randn(
            token.shape,
            device=token.device,
            dtype=token.dtype,
            generator=generator,
        )
    return spatial, token


@torch.no_grad()
def prepare_attacker_inputs(
    semantic_model: GlobalSemanticBottleneck,
    token: Tensor,
    labels: Tensor,
    attack_view: str,
    identity_conditioning: str,
) -> tuple[Tensor, Tensor | None, Tensor]:
    predicted_posterior = semantic_model.classifier(token).softmax(dim=1)
    if attack_view == "spatial":
        attacker_token = torch.zeros_like(token)
    elif attack_view == "joint":
        attacker_token = token
    else:
        raise ValueError(f"unsupported attack view: {attack_view}")

    if identity_conditioning == "none":
        identity_condition = None
    elif identity_conditioning == "uniform":
        identity_condition = torch.full_like(
            predicted_posterior, 1.0 / predicted_posterior.shape[1]
        )
    elif identity_conditioning == "predicted":
        if attack_view != "joint":
            raise ValueError("predicted identity conditioning requires the semantic path")
        identity_condition = predicted_posterior
    elif identity_conditioning == "oracle":
        identity_condition = F.one_hot(labels, num_classes=predicted_posterior.shape[1]).to(
            dtype=token.dtype
        )
    else:
        raise ValueError(f"unsupported identity conditioning: {identity_conditioning}")
    return attacker_token, identity_condition, predicted_posterior


def reconstruct(
    generator_model: nn.Module,
    spatial: Tensor,
    token: Tensor,
    identity_condition: Tensor | None,
) -> Tensor:
    if isinstance(generator_model, IdentityConditionedDualPathReconstructor):
        if identity_condition is None:
            raise ValueError("conditioned reconstructor requires an identity condition")
        return generator_model(spatial, token, identity_condition)
    if identity_condition is not None:
        raise ValueError("unconditioned reconstructor received an identity condition")
    return generator_model(spatial, token)


@torch.no_grad()
def evaluate(
    generator_model: nn.Module,
    legacy: nn.Module,
    semantic: GlobalSemanticBottleneck,
    loader: DataLoader,
    device: torch.device,
    legacy_noise_std: float,
    semantic_noise_std: float,
    seed: int,
    attack_view: str = "joint",
    identity_conditioning: str = "none",
    lpips_model=None,
    identity_model=None,
) -> dict:
    generator_model.eval()
    random_generator = torch.Generator(device=device).manual_seed(seed)
    absolute_error = 0.0
    metric_totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    if lpips_model is not None:
        metric_totals["lpips"] = 0.0
    if identity_model is not None:
        metric_totals["identity_cosine_similarity"] = 0.0
    semantic_identity_correct = 0
    reconstructed_identity_correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spatial, token = transmitted_features(
            legacy,
            semantic,
            images,
            legacy_noise_std,
            semantic_noise_std,
            random_generator,
        )
        attacker_token, identity_condition, predicted_posterior = prepare_attacker_inputs(
            semantic,
            token,
            labels,
            attack_view,
            identity_conditioning,
        )
        reconstructed = reconstruct(
            generator_model, spatial, attacker_token, identity_condition
        )
        semantic_identity_correct += int((predicted_posterior.argmax(dim=1) == labels).sum())
        reconstructed_identity_logits = semantic.classifier(semantic.encode(reconstructed))
        reconstructed_identity_correct += int(
            (reconstructed_identity_logits.argmax(dim=1) == labels).sum()
        )
        per_image = reconstruction_metrics_per_image(reconstructed, images)
        if lpips_model is not None:
            per_image["lpips"] = lpips_model(
                reconstructed.mul(2).sub(1), images.mul(2).sub(1)
            ).flatten()
        if identity_model is not None:
            reconstructed_face = F.interpolate(
                reconstructed, size=(160, 160), mode="bilinear", align_corners=False
            )
            original_face = F.interpolate(
                images, size=(160, 160), mode="bilinear", align_corners=False
            )
            reconstructed_embedding = F.normalize(
                identity_model(reconstructed_face.sub(0.5).div(0.5)), dim=1
            )
            original_embedding = F.normalize(
                identity_model(original_face.sub(0.5).div(0.5)), dim=1
            )
            per_image["identity_cosine_similarity"] = (
                reconstructed_embedding * original_embedding
            ).sum(dim=1)
        batch_size = images.shape[0]
        for name in metric_totals:
            metric_totals[name] += float(per_image[name].sum())
        absolute_error += float(
            (reconstructed - images).abs().flatten(start_dim=1).mean(dim=1).sum()
        )
        count += batch_size
    if count == 0:
        raise ValueError("attack evaluation loader is empty")
    return {
        **{name: total / count for name, total in metric_totals.items()},
        "mae": absolute_error / count,
        "semantic_identity_top1_accuracy": semantic_identity_correct / count,
        "reconstructed_identity_top1_accuracy": reconstructed_identity_correct / count,
        "images": count,
    }


@torch.no_grad()
def evaluate_over_noise_seeds(
    generator_model: nn.Module,
    legacy: nn.Module,
    semantic: GlobalSemanticBottleneck,
    loader: DataLoader,
    device: torch.device,
    legacy_noise_std: float,
    semantic_noise_std: float,
    seeds: list[int],
    attack_view: str = "joint",
    identity_conditioning: str = "none",
    lpips_model=None,
    identity_model=None,
) -> dict:
    if not seeds:
        raise ValueError("at least one evaluation noise seed is required")
    runs = [
        {
            "noise_seed": seed,
            **evaluate(
                generator_model,
                legacy,
                semantic,
                loader,
                device,
                legacy_noise_std,
                semantic_noise_std,
                seed,
                attack_view,
                identity_conditioning,
                lpips_model,
                identity_model,
            ),
        }
        for seed in seeds
    ]
    metric_names = [
        "mse",
        "mae",
        "psnr",
        "ssim",
        "semantic_identity_top1_accuracy",
        "reconstructed_identity_top1_accuracy",
    ]
    if lpips_model is not None:
        metric_names.append("lpips")
    if identity_model is not None:
        metric_names.append("identity_cosine_similarity")
    aggregate = {
        name: float(np.mean([float(run[name]) for run in runs])) for name in metric_names
    }
    aggregate["images"] = int(runs[0]["images"])
    aggregate["noise_seed_runs"] = runs
    aggregate["noise_seed_count"] = len(runs)
    if len(runs) > 1:
        aggregate["noise_seed_standard_deviation"] = {
            name: float(np.std([float(run[name]) for run in runs], ddof=1))
            for name in metric_names
        }
    return aggregate


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to append to existing run: {log_path}")
    train_loader, evaluation_loader = build_attack_loaders(
        args.data_root,
        args.batch_size,
        args.workers,
        args.attack_knowledge,
        args.max_auxiliary_samples,
        args.max_evaluation_samples,
    )
    (
        legacy,
        semantic,
        legacy_noise,
        semantic_noise,
        token_dim,
        checkpoint_class_to_index,
    ) = load_feature_extractors(args.dual_path_checkpoint, device, args.legacy_checkpoint_dir)
    if dataset_class_to_index(train_loader) != checkpoint_class_to_index:
        raise ValueError("attack data and target checkpoint class mappings differ")
    if dataset_class_to_index(evaluation_loader) != checkpoint_class_to_index:
        raise ValueError("evaluation data and target checkpoint class mappings differ")
    legacy_noise = resolve_noise_std(legacy_noise, args.legacy_noise_std)
    semantic_noise = resolve_noise_std(semantic_noise, args.semantic_noise_std)
    if args.identity_conditioning == "none":
        reconstructor = DualPathReconstructor(
            semantic_dim=token_dim,
            width=args.width,
            residual_blocks=args.residual_blocks,
        ).to(device)
    else:
        reconstructor = IdentityConditionedDualPathReconstructor(
            semantic_dim=token_dim,
            num_identities=semantic.classifier.out_features,
            width=args.width,
            residual_blocks=args.residual_blocks,
        ).to(device)
    if args.initial_decoder_checkpoint:
        if args.identity_conditioning != "none":
            raise ValueError(
                "an unconditioned decoder checkpoint cannot initialise a "
                "capacity-matched identity-conditioned attack"
            )
        initial = torch.load(
            args.initial_decoder_checkpoint, map_location="cpu", weights_only=False
        )
        reconstructor.load_state_dict(initial["reconstructor_state"])

    discriminator = None
    discriminator_optimizer = None
    if args.attack_type == "gan":
        discriminator = PatchDiscriminator().to(device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            betas=(0.5, 0.999),
        )
    generator_optimizer = torch.optim.AdamW(
        reconstructor.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optimizer, T_max=args.epochs, eta_min=1e-5
    )
    adversarial_criterion = nn.BCEWithLogitsLoss()
    best_mse = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        reconstructor.train()
        if discriminator is not None:
            discriminator.train()
        totals = {"reconstruction": 0.0, "generator_adv": 0.0, "count": 0}
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spatial, token = transmitted_features(
                legacy,
                semantic,
                images,
                legacy_noise,
                semantic_noise,
            )
            attacker_token, identity_condition, _predicted_posterior = prepare_attacker_inputs(
                semantic,
                token,
                labels,
                args.attack_view,
                args.identity_conditioning,
            )
            reconstructed = reconstruct(
                reconstructor, spatial, attacker_token, identity_condition
            )

            if discriminator is not None and discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_scores = discriminator(images)
                fake_scores = discriminator(reconstructed.detach())
                discriminator_loss = 0.5 * (
                    adversarial_criterion(real_scores, torch.ones_like(real_scores))
                    + adversarial_criterion(fake_scores, torch.zeros_like(fake_scores))
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()

            reconstruction_loss = F.mse_loss(reconstructed, images)
            generator_adv = reconstruction_loss.detach() * 0.0
            if discriminator is not None:
                fake_scores = discriminator(reconstructed)
                generator_adv = adversarial_criterion(fake_scores, torch.ones_like(fake_scores))
            generator_loss = reconstruction_loss + args.gan_weight * generator_adv
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=5.0)
            generator_optimizer.step()

            batch_size = images.shape[0]
            totals["reconstruction"] += float(reconstruction_loss.detach()) * batch_size
            totals["generator_adv"] += float(generator_adv.detach()) * batch_size
            totals["count"] += batch_size

        generator_scheduler.step()
        evaluation = evaluate(
            reconstructor,
            legacy,
            semantic,
            evaluation_loader,
            device,
            legacy_noise,
            semantic_noise,
            args.seed + 100_000,
            args.attack_view,
            args.identity_conditioning,
        )
        record = {
            "epoch": epoch,
            "train_reconstruction_loss": totals["reconstruction"] / totals["count"],
            "train_generator_adversarial_loss": totals["generator_adv"] / totals["count"],
            "evaluation_mse": evaluation["mse"],
            "evaluation_psnr": evaluation["psnr"],
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if evaluation["mse"] < best_mse:
            best_mse = evaluation["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "reconstructor_state": reconstructor.state_dict(),
                    "evaluation": evaluation,
                    "args": vars(args),
                },
                args.output_dir / "checkpoint_best.pt",
            )

    best = torch.load(
        args.output_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    reconstructor.load_state_dict(best["reconstructor_state"])
    lpips_model = None
    identity_model = None
    if args.perceptual_metrics:
        import lpips
        from facenet_pytorch import InceptionResnetV1

        lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
        identity_model = (
            InceptionResnetV1(pretrained="vggface2").to(device).eval().requires_grad_(False)
        )
    final_noise_seeds = args.evaluation_noise_seeds or [args.seed + 300_000]
    evaluation_metrics = evaluate_over_noise_seeds(
        reconstructor,
        legacy,
        semantic,
        evaluation_loader,
        device,
        legacy_noise,
        semantic_noise,
        final_noise_seeds,
        args.attack_view,
        args.identity_conditioning,
        lpips_model,
        identity_model,
    )
    result = {
        "attack_type": args.attack_type,
        "attack_view": args.attack_view,
        "identity_conditioning": args.identity_conditioning,
        "attack_knowledge": args.attack_knowledge,
        "best_epoch": best_epoch,
        "evaluation": evaluation_metrics,
        "dual_path_checkpoint": str(args.dual_path_checkpoint),
        "attacker_observes_spatial_and_semantic_paths": args.attack_view == "joint",
        "oracle_identity_is_upper_bound_only": args.identity_conditioning == "oracle",
        "effective_legacy_noise_std": legacy_noise,
        "effective_semantic_noise_std": semantic_noise,
        "evaluation_noise_seeds": final_noise_seeds,
        "perceptual_metrics_enabled": args.perceptual_metrics,
        "identity_evaluator": "InceptionResnetV1 pretrained on VGGFace2"
        if args.perceptual_metrics
        else None,
        "perceptual_evaluator": "LPIPS-Alex" if args.perceptual_metrics else None,
    }
    save_json(args.output_dir / "attack_metrics.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
