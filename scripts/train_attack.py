#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_imagefolder_loaders, build_protocol_loaders
from publication_cem.evaluators import FaceIdentityEvaluator, LPIPSMetric
from publication_cem.metrics import (
    MetricAccumulator,
    reconstruction_metrics,
    reconstruction_metrics_per_image,
)
from publication_cem.models import (
    PatchDiscriminator,
    build_inversion_decoder,
    build_split_model,
)
from publication_cem.reproducibility import (
    append_jsonl,
    module_state_sha256,
    save_json,
    seed_everything,
)


ATTACK_TYPES = ("conv_decoder", "residual_decoder", "gan", "adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attack-type", choices=ATTACK_TYPES)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="stop after auxiliary validation; never read the target-test loader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override the checkpoint's default attacker seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="override attack epochs without changing the frozen target checkpoint",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="override the attack learning rate",
    )
    parser.add_argument(
        "--auxiliary-root",
        type=Path,
        help="override the protocol auxiliary split with an external ImageFolder root",
    )
    return parser.parse_args()


def transmitted_feature(
    model, images, noise_std: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    with torch.no_grad():
        return model.transmit(model.encode(images), noise_std, generator)


def reconstruction_loss(
    decoder,
    model,
    images,
    noise_std: float,
    attack_type: str,
    eot_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attack_type != "adaptive":
        reconstruction = decoder(transmitted_feature(model, images, noise_std))
        return F.mse_loss(reconstruction, images), reconstruction

    losses = []
    reconstructions = []
    for _ in range(eot_samples):
        reconstruction = decoder(transmitted_feature(model, images, noise_std))
        losses.append(F.mse_loss(reconstruction, images))
        reconstructions.append(reconstruction)
    return torch.stack(losses).mean(), torch.stack(reconstructions).mean(dim=0)


@torch.no_grad()
def evaluate_auxiliary(
    decoder,
    model,
    loader,
    device,
    noise_std: float,
    evaluation_seed: int,
) -> float:
    decoder.eval()
    total = 0.0
    count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(evaluation_seed)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction = decoder(
            transmitted_feature(model, images, noise_std, generator)
        )
        total += float(F.mse_loss(reconstruction, images)) * images.shape[0]
        count += images.shape[0]
    if not count:
        raise ValueError("attacker auxiliary validation loader is empty")
    return total / count


def convergence_record(history: list[float]) -> tuple[bool, float | None]:
    if len(history) < 10:
        return False, None
    previous = sum(history[-10:-5]) / 5
    recent = sum(history[-5:]) / 5
    relative_change = abs(previous - recent) / max(abs(previous), 1e-12)
    return relative_change < 0.01, relative_change


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    attack_cfg = dict(config["attack"])
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("attack epochs must be positive")
        attack_cfg["epochs"] = args.epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("attack learning rate must be positive")
        attack_cfg["learning_rate"] = args.learning_rate
    attack_seed = (
        int(args.seed)
        if args.seed is not None
        else int(attack_cfg.get("seed", config["seed"] + 10_000))
    )
    attack_type = args.attack_type or attack_cfg.get("type", "conv_decoder")
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unsupported attack type: {attack_type}")
    if args.auxiliary_root is not None:
        attack_cfg["auxiliary_root"] = str(args.auxiliary_root)
    seed_everything(attack_seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "attack_training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to append to existing attack log: {log_path}")

    data_cfg = config["data"]
    target_loaders = build_protocol_loaders(
        data_cfg,
        attack_seed,
        batch_size=int(attack_cfg.get("batch_size", data_cfg["batch_size"])),
    )
    if target_loaders.class_to_index != checkpoint["class_to_index"]:
        raise ValueError("dataset class mapping differs from the target checkpoint")

    auxiliary_root = attack_cfg.get("auxiliary_root")
    if auxiliary_root:
        auxiliary_loaders = build_imagefolder_loaders(
            data_root=auxiliary_root,
            image_size=int(data_cfg["image_size"]),
            batch_size=int(attack_cfg.get("batch_size", data_cfg["batch_size"])),
            workers=int(data_cfg.get("workers", 4)),
            seed=attack_seed,
        )
        auxiliary_train_loader = auxiliary_loaders.train
        auxiliary_validation_loader = auxiliary_loaders.validation
        auxiliary_split = "external_train_validation"
    else:
        auxiliary_train_loader = target_loaders.attacker_auxiliary_train
        auxiliary_validation_loader = target_loaders.attacker_auxiliary_validation
        auxiliary_split = (
            "protocol_attacker_auxiliary"
            if target_loaders.split_manifest is not None
            or target_loaders.protocol_mode == "cem_facescrub_paper"
            else "legacy_target_train_validation"
        )

    model_cfg = config["model"]
    checkpoint_model_cfg = dict(model_cfg)
    checkpoint_model_cfg["pretrained"] = False
    model = build_split_model(
        checkpoint_model_cfg,
        num_classes=target_loaders.num_classes,
        image_size=int(data_cfg["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.requires_grad_(False)

    defense_cfg = config["defense"]
    noise_std = 0.0 if defense_cfg["name"] == "none" else float(defense_cfg["noise_std"])
    decoder_width = int(attack_cfg.get("decoder_width", 128))
    decoder = build_inversion_decoder(
        attack_type,
        input_channels=int(checkpoint["smashed_shape"][0]),
        output_size=int(data_cfg["image_size"]),
        width=decoder_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(attack_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(attack_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(attack_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    eot_samples = int(attack_cfg.get("adaptive_eot_samples", 4))
    gan_weight = float(attack_cfg.get("gan_weight", 0.01))
    discriminator = None
    discriminator_optimizer = None
    if attack_type == "gan":
        discriminator = PatchDiscriminator(max(16, decoder_width // 2)).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(attack_cfg.get("gan_discriminator_learning_rate", 2e-4)),
            betas=(0.5, 0.999),
        )

    best_validation_mse = math.inf
    best_epoch = 0
    validation_history: list[float] = []
    best_decoder_path = args.output_dir / "decoder_best.pt"
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        decoder.train()
        loss_total = 0.0
        reconstruction_total = 0.0
        adversarial_total = 0.0
        discriminator_total = 0.0
        count = 0
        for images, _ in auxiliary_train_loader:
            images = images.to(device, non_blocking=True)
            reconstruction_mse, reconstruction = reconstruction_loss(
                decoder,
                model,
                images,
                noise_std,
                attack_type,
                eot_samples,
            )
            adversarial_loss = reconstruction_mse.detach() * 0.0
            discriminator_loss = reconstruction_mse.detach() * 0.0
            if discriminator is not None and discriminator_optimizer is not None:
                discriminator.train()
                real_logits = discriminator(images)
                fake_logits = discriminator(reconstruction.detach())
                discriminator_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        real_logits, torch.ones_like(real_logits)
                    )
                    + F.binary_cross_entropy_with_logits(
                        fake_logits, torch.zeros_like(fake_logits)
                    )
                )
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss.backward()
                discriminator_optimizer.step()
                adversarial_logits = discriminator(reconstruction)
                adversarial_loss = F.binary_cross_entropy_with_logits(
                    adversarial_logits, torch.ones_like(adversarial_logits)
                )

            loss = reconstruction_mse + gan_weight * adversarial_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = images.shape[0]
            loss_total += float(loss.detach()) * batch_size
            reconstruction_total += float(reconstruction_mse.detach()) * batch_size
            adversarial_total += float(adversarial_loss.detach()) * batch_size
            discriminator_total += float(discriminator_loss.detach()) * batch_size
            count += batch_size

        scheduler.step()
        validation_mse = evaluate_auxiliary(
            decoder,
            model,
            auxiliary_validation_loader,
            device,
            noise_std,
            evaluation_seed=attack_seed + 100_000,
        )
        validation_history.append(validation_mse)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            torch.save(decoder.state_dict(), best_decoder_path)
        converged, relative_change = convergence_record(validation_history)
        record = {
            "epoch": epoch,
            "attack_type": attack_type,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_total_loss": loss_total / count,
            "train_mse": reconstruction_total / count,
            "train_adversarial_loss": adversarial_total / count,
            "train_discriminator_loss": discriminator_total / count,
            "auxiliary_validation_mse": validation_mse,
            "best_auxiliary_validation_mse": best_validation_mse,
            "best_epoch": best_epoch,
            "converged_last_10": converged,
            "convergence_relative_change": relative_change,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    decoder.load_state_dict(torch.load(best_decoder_path, map_location=device, weights_only=True))
    decoder.eval()
    converged, relative_change = convergence_record(validation_history)
    if args.selection_only:
        selection_metrics = {
            "checkpoint": str(args.checkpoint),
            "attack_type": attack_type,
            "attack_seed": attack_seed,
            "attack_epochs": epochs,
            "best_epoch": best_epoch,
            "best_auxiliary_validation_mse": best_validation_mse,
            "converged_last_10": converged,
            "convergence_relative_change": relative_change,
            "auxiliary_split": auxiliary_split,
            "evaluation_split": "attacker_auxiliary_validation",
            "evaluation_noise_seed": attack_seed + 100_000,
            "target_test_accessed": False,
        }
        save_json(
            args.output_dir / "attack_selection_metrics.json", selection_metrics
        )
        torch.save(decoder.state_dict(), args.output_dir / "decoder.pt")
        if discriminator is not None:
            torch.save(
                discriminator.state_dict(), args.output_dir / "discriminator.pt"
            )
        print(json.dumps(selection_metrics, sort_keys=True))
        return

    evaluation_cfg = attack_cfg.get("evaluation", {})
    lpips_cfg = evaluation_cfg.get("lpips", {})
    face_cfg = evaluation_cfg.get("face_identity", {})
    lpips_metric = None
    face_evaluator = None
    if lpips_cfg.get("enabled", False):
        lpips_metric = LPIPSMetric(
            device,
            net=lpips_cfg.get("net", "alex"),
            version=lpips_cfg.get("version", "0.1"),
        )
        assets_path = Path(evaluation_cfg["assets_manifest"])
        if not assets_path.is_absolute():
            assets_path = ROOT / assets_path
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        if (
            assets.get("status") != "VERIFIED"
            or assets.get("lpips_net") != lpips_cfg.get("net", "alex")
            or assets.get("lpips_version") != lpips_cfg.get("version", "0.1")
            or assets.get("lpips_weights_sha256")
            != module_state_sha256(lpips_metric.model)
        ):
            raise ValueError("loaded LPIPS model differs from evaluation asset manifest")
    face_protocol_path = None
    if face_cfg.get("enabled", False):
        face_protocol_path = Path(face_cfg["protocol"])
        if not face_protocol_path.is_absolute():
            face_protocol_path = ROOT / face_protocol_path
        if not face_protocol_path.is_file():
            raise FileNotFoundError(
                f"required face identity protocol is missing: {face_protocol_path}"
            )
        face_evaluator = FaceIdentityEvaluator.from_protocol(
            face_protocol_path,
            device,
            checkpoint["class_to_index"],
            expected_target_far=float(face_cfg.get("target_far", 0.001)),
        )

    accumulator = MetricAccumulator()
    extended_totals = {
        "lpips": 0.0,
        "identity_top1_success": 0.0,
        "face_cosine_similarity": 0.0,
        "true_identity_prototype_similarity": 0.0,
        "verification_tar_at_far": 0.0,
    }
    extended_count = {name: 0 for name in extended_totals}
    per_image_path = args.output_dir / "per_image_metrics.jsonl"
    per_image_path.write_text("", encoding="utf-8")
    image_index = 0
    target_evaluation_seed = int(config["seed"]) + 300_000
    target_evaluation_generator = torch.Generator(device=device)
    target_evaluation_generator.manual_seed(target_evaluation_seed)
    test_identifiers = []
    if target_loaders.split_manifest is not None:
        test_record = target_loaders.split_manifest["splits"]["target_test"]
        if "paths" in test_record:
            test_identifiers = list(test_record["paths"])
        elif "indices" in test_record:
            test_identifiers = [f"test:{index}" for index in test_record["indices"]]
    with torch.no_grad():
        for images, labels in target_loaders.test:
            images = images.to(device, non_blocking=True)
            reconstruction = decoder(
                transmitted_feature(
                    model,
                    images,
                    noise_std,
                    target_evaluation_generator,
                )
            )
            metrics = reconstruction_metrics(reconstruction, images)
            accumulator.update(metrics, images.shape[0])
            per_image = reconstruction_metrics_per_image(reconstruction, images)
            lpips_scores = (
                lpips_metric(reconstruction, images) if lpips_metric is not None else None
            )
            identity = (
                face_evaluator(reconstruction, images, labels)
                if face_evaluator is not None
                else None
            )
            if lpips_scores is not None:
                extended_totals["lpips"] += float(lpips_scores.sum().cpu())
                extended_count["lpips"] += images.shape[0]
            if identity is not None:
                identity_values = {
                    "identity_top1_success": identity.top1_success,
                    "face_cosine_similarity": identity.original_reconstruction_cosine,
                    "true_identity_prototype_similarity": (
                        identity.true_prototype_similarity
                    ),
                    "verification_tar_at_far": identity.verification_accept,
                }
                for name, values in identity_values.items():
                    extended_totals[name] += float(values.sum().cpu())
                    extended_count[name] += images.shape[0]
            for row in range(images.shape[0]):
                row_metrics = {
                    "index": image_index,
                    "sample_id": (
                        test_identifiers[image_index] if test_identifiers else None
                    ),
                    "label": int(labels[row]),
                    "mse": float(per_image["mse"][row].cpu()),
                    "ssim": float(per_image["ssim"][row].cpu()),
                    "psnr": float(per_image["psnr"][row].cpu()),
                }
                if lpips_scores is not None:
                    row_metrics["lpips"] = float(lpips_scores[row].cpu())
                if identity is not None:
                    row_metrics.update(
                        identity_top1_success=float(
                            identity.top1_success[row].cpu()
                        ),
                        face_cosine_similarity=float(
                            identity.original_reconstruction_cosine[row].cpu()
                        ),
                        true_identity_prototype_similarity=float(
                            identity.true_prototype_similarity[row].cpu()
                        ),
                        verification_accept_at_far=float(
                            identity.verification_accept[row].cpu()
                        ),
                    )
                append_jsonl(
                    per_image_path,
                    row_metrics,
                )
                image_index += 1
    final_metrics = accumulator.compute()
    for name, total in extended_totals.items():
        if extended_count[name]:
            final_metrics[name] = total / extended_count[name]
    final_metrics.update(
        checkpoint=str(args.checkpoint),
        attack_type=attack_type,
        attack_seed=attack_seed,
        attack_epochs=epochs,
        best_epoch=best_epoch,
        best_auxiliary_validation_mse=best_validation_mse,
        converged_last_10=converged,
        convergence_relative_change=relative_change,
        auxiliary_split=auxiliary_split,
        evaluation_split=(
            "target_test"
            if target_loaders.split_manifest is not None
            or target_loaders.protocol_mode == "cem_facescrub_paper"
            else "legacy_validation"
        ),
        evaluation_noise_seed=target_evaluation_seed,
        lpips_enabled=lpips_metric is not None,
        face_identity_enabled=face_evaluator is not None,
        face_identity_protocol=(
            str(face_protocol_path) if face_protocol_path is not None else None
        ),
        face_identity_target_far=(
            float(face_cfg.get("target_far", 0.001))
            if face_evaluator is not None
            else None
        ),
    )
    save_json(args.output_dir / "attack_metrics.json", final_metrics)
    torch.save(decoder.state_dict(), args.output_dir / "decoder.pt")
    if discriminator is not None:
        torch.save(discriminator.state_dict(), args.output_dir / "discriminator.pt")
    print(json.dumps(final_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
