#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.config import PrototypeCEMConfig
from publication_cem.data import build_protocol_loaders
from publication_cem.epochwise_cem import (
    EpochwiseCEMConfig,
    EpochwiseKMeansCEMRegularizer,
)
from publication_cem.models import build_split_model
from publication_cem.objective import PrivacyUtilityObjective
from publication_cem.official_cem import OfficialCEMConfig, OfficialCEMRegularizer
from publication_cem.regularizer import PrototypeCEMRegularizer
from publication_cem.reproducibility import (
    append_jsonl,
    save_json,
    seed_everything,
    tensor_sha256,
)
from publication_cem.training_control import (
    scheduled_privacy_weight,
    task_priority_gradient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="select by target validation and never read target test",
    )
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    noise_std: float,
    evaluation_seed: int,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    count = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device)
    generator.manual_seed(evaluation_seed)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smashed = model.encode(images)
            transmitted = model.transmit(smashed, noise_std, generator)
            logits = model.decode(transmitted)
            loss = criterion(logits, labels)
            loss_total += float(loss) * images.shape[0]
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += images.shape[0]
    return {"loss": loss_total / count, "accuracy": correct / count}


@torch.no_grad()
def refresh_class_statistics(
    model: nn.Module,
    loader,
    regularizer: nn.Module,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    started = time.perf_counter()
    model.eval()
    grouped = [[] for _ in range(num_classes)]
    feature_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = model.encode(images).flatten(start_dim=1).cpu()
        labels = labels.reshape(-1)
        for class_tensor in torch.unique(labels):
            class_index = int(class_tensor.item())
            grouped[class_index].append(features[labels == class_tensor])
        feature_count += features.shape[0]
    for class_index, chunks in enumerate(grouped):
        if chunks:
            regularizer.fit_class(class_index, torch.cat(chunks, dim=0))
    if isinstance(regularizer, OfficialCEMRegularizer):
        regularizer.complete_refresh()
    return {
        "statistics_refresh_seconds": time.perf_counter() - started,
        "statistics_feature_count": feature_count,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    data_cfg = config["data"]
    train_cfg = config["training"]
    loaders = build_protocol_loaders(data_cfg, seed)
    model_cfg = config["model"]
    model = build_split_model(
        model_cfg,
        num_classes=loaders.num_classes,
        image_size=int(data_cfg["image_size"]),
    ).to(device)
    initialization_record = {
        "initialization_checkpoint": None,
        "initialization_missing_keys": [],
        "initialization_unexpected_keys": [],
    }
    initialization_checkpoint = train_cfg.get("initialization_checkpoint")
    if initialization_checkpoint:
        initialization_path = Path(initialization_checkpoint)
        if not initialization_path.is_absolute():
            initialization_path = ROOT / initialization_path
        initial_state = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        incompatible = model.load_state_dict(initial_state["model_state"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        allowed_missing_prefixes = ("channel_noise.", "margin_head.")
        if unexpected or any(
            not key.startswith(allowed_missing_prefixes) for key in missing
        ):
            raise ValueError(
                "initialization checkpoint is incompatible with the target model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        initialization_record = {
            "initialization_checkpoint": str(initialization_path),
            "initialization_missing_keys": missing,
            "initialization_unexpected_keys": unexpected,
        }

    distillation_cfg = dict(train_cfg.get("distillation", {}))
    distillation_weight = float(distillation_cfg.get("weight", 0.0))
    distillation_temperature = float(distillation_cfg.get("temperature", 2.0))
    if distillation_weight < 0 or distillation_temperature <= 0:
        raise ValueError("invalid distillation weight or temperature")
    teacher = None
    teacher_checkpoint_path = distillation_cfg.get("checkpoint")
    if distillation_weight > 0:
        if not teacher_checkpoint_path:
            raise ValueError("positive distillation weight requires a teacher checkpoint")
        teacher_path = Path(teacher_checkpoint_path)
        if not teacher_path.is_absolute():
            teacher_path = ROOT / teacher_path
        teacher_checkpoint = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        if teacher_checkpoint["class_to_index"] != loaders.class_to_index:
            raise ValueError("teacher and student class mappings differ")
        teacher_model_cfg = dict(teacher_checkpoint["config"]["model"])
        teacher_model_cfg["pretrained"] = False
        teacher = build_split_model(
            teacher_model_cfg,
            num_classes=loaders.num_classes,
            image_size=int(data_cfg["image_size"]),
        ).to(device)
        teacher.load_state_dict(teacher_checkpoint["model_state"])
        teacher.eval().requires_grad_(False)
        teacher_checkpoint_path = str(teacher_path)

    trainable_model_parts = train_cfg.get("trainable_model_parts")
    if trainable_model_parts is not None:
        if not isinstance(trainable_model_parts, list) or not trainable_model_parts:
            raise ValueError("trainable_model_parts must be a non-empty list")
        model.requires_grad_(False)
        for part_name in trainable_model_parts:
            part = getattr(model, part_name, None)
            if not isinstance(part, nn.Module):
                raise ValueError(f"unknown or disabled trainable model part: {part_name}")
            part.requires_grad_(True)
    smashed_shape = model.smashed_shape((3, data_cfg["image_size"], data_cfg["image_size"]))
    feature_dim = int(torch.tensor(smashed_shape).prod().item())

    defense_cfg = config["defense"]
    defense_name = defense_cfg["name"]
    noise_std = 0.0 if defense_name == "none" else float(defense_cfg["noise_std"])
    objective: Optional[PrivacyUtilityObjective] = None
    regularizer: Optional[nn.Module] = None
    if defense_name == "prototype_cem":
        regularizer_options = dict(defense_cfg["regularizer"])
        regularizer_options.update(
            feature_dim=feature_dim,
            num_classes=loaders.num_classes,
            noise_std=noise_std,
            seed=seed,
        )
        regularizer = PrototypeCEMRegularizer(
            PrototypeCEMConfig(**regularizer_options)
        ).to(device)
        objective = PrivacyUtilityObjective(
            regularizer, float(defense_cfg["privacy_weight"])
        )
    elif defense_name == "epochwise_cem":
        epochwise_options = dict(defense_cfg["epochwise"])
        epochwise_options.update(
            feature_dim=feature_dim,
            num_classes=loaders.num_classes,
            noise_std=noise_std,
        )
        regularizer = EpochwiseKMeansCEMRegularizer(
            EpochwiseCEMConfig(**epochwise_options)
        ).to(device)
        objective = PrivacyUtilityObjective(
            regularizer, float(defense_cfg["privacy_weight"])
        )
    elif defense_name == "official_cem":
        official_options = dict(defense_cfg["official"])
        official_options.update(
            feature_dim=feature_dim,
            num_classes=loaders.num_classes,
            dataset_size=len(loaders.statistics.dataset),
            noise_std=noise_std,
            seed=seed,
        )
        regularizer = OfficialCEMRegularizer(
            OfficialCEMConfig(**official_options)
        ).to(device)
        objective = PrivacyUtilityObjective(
            regularizer, float(defense_cfg["privacy_weight"])
        )
    elif defense_name not in {"none", "gaussian"}:
        raise ValueError(f"unsupported defense: {defense_name}")
    projector = (
        regularizer.projector
        if isinstance(regularizer, PrototypeCEMRegularizer)
        else None
    )

    optimization_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if regularizer is not None:
        optimization_parameters.extend(
            parameter for parameter in regularizer.parameters() if parameter.requires_grad
        )
    if not optimization_parameters:
        raise ValueError("the training configuration has no trainable parameters")
    optimizer_name = str(train_cfg.get("optimizer", "sgd"))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            optimization_parameters,
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            optimization_parameters,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError("training.optimizer must be 'sgd' or 'adamw'")
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if not 0 <= label_smoothing < 1:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
    gradient_diagnostic_interval = int(
        train_cfg.get("gradient_diagnostic_interval", 50)
    )
    gradient_composition = str(train_cfg.get("gradient_composition", "sum"))
    if gradient_composition not in {"sum", "task_priority"}:
        raise ValueError("gradient_composition must be 'sum' or 'task_priority'")
    privacy_schedule = dict(train_cfg.get("privacy_weight_schedule", {}))
    base_privacy_weight = float(defense_cfg.get("privacy_weight", 0.0))
    margin_weight = float(train_cfg.get("margin_weight", 0.0))
    if margin_weight < 0:
        raise ValueError("margin_weight must be non-negative")
    if margin_weight > 0 and model.margin_head is None:
        raise ValueError("margin_weight requires model.margin_head.enabled=true")
    best_accuracy = -1.0
    log_path = output_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(
            f"refusing to append to existing run log: {log_path}; use a new output_dir"
        )

    if initialization_checkpoint:
        initial_validation = evaluate(
            model,
            loaders.validation,
            device,
            noise_std,
            evaluation_seed=seed + 100_000,
        )
        initial_projection_sha256 = (
            tensor_sha256(projector.matrix) if projector is not None else None
        )
        initial_checkpoint = {
            "epoch": 0,
            "model_state": model.state_dict(),
            "regularizer_state": regularizer.state_dict() if regularizer else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "class_to_index": loaders.class_to_index,
            "smashed_shape": smashed_shape,
            "config": config,
            "validation": initial_validation,
            "assignment_projection_sha256": initial_projection_sha256,
        }
        torch.save(initial_checkpoint, output_dir / "checkpoint_best.pt")
        best_accuracy = initial_validation["accuracy"]
        save_json(
            output_dir / "initial_validation_metrics.json",
            {
                "epoch": 0,
                "validation_loss": initial_validation["loss"],
                "validation_accuracy": initial_validation["accuracy"],
                "evaluation_noise_seed": seed + 100_000,
                "selection_split": "target_validation",
                "target_test_accessed": False,
                **initialization_record,
            },
        )

    for epoch in range(1, epochs + 1):
        total_epoch_started = time.perf_counter()
        effective_privacy_weight = scheduled_privacy_weight(
            base_privacy_weight,
            epoch,
            warmup_epochs=int(privacy_schedule.get("warmup_epochs", 0)),
            ramp_epochs=int(privacy_schedule.get("ramp_epochs", 0)),
            warmup_factor=float(privacy_schedule.get("warmup_factor", 0.0)),
        )
        if objective is not None:
            objective.privacy_weight = effective_privacy_weight
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        refresh_record = {
            "statistics_refresh_seconds": 0.0,
            "statistics_feature_count": 0,
        }
        if isinstance(regularizer, EpochwiseKMeansCEMRegularizer):
            refresh_record = refresh_class_statistics(
                model,
                loaders.statistics,
                regularizer,
                loaders.num_classes,
                device,
            )
        model.train()
        if trainable_model_parts is not None:
            for part_name, part in model.named_children():
                if part_name not in trainable_model_parts:
                    part.eval()
        training_started = time.perf_counter()
        if regularizer is not None:
            regularizer.train()
        totals = {
            "classification": 0.0,
            "margin": 0.0,
            "distillation": 0.0,
            "task": 0.0,
            "privacy": 0.0,
            "total": 0.0,
            "count": 0,
        }
        diagnostic_totals = {}
        diagnostic_last = {}

        gradient_diagnostic_batches = 0
        gradient_task_norm_total = 0.0
        gradient_privacy_norm_total = 0.0
        gradient_conflict_fraction_total = 0.0
        gradient_cosine_total = 0.0
        gradient_composition_batches = 0
        for batch_index, (images, labels) in enumerate(loaders.train):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smashed = model.encode(images)
            if not smashed.requires_grad:
                smashed = smashed.detach().requires_grad_(True)
            transmitted = model.transmit(smashed, noise_std)
            logits = model.decode(transmitted)
            classification_loss = criterion(logits, labels)
            margin_loss = classification_loss.detach() * 0.0
            if margin_weight > 0:
                margin_loss = F.cross_entropy(
                    model.auxiliary_margin_logits(smashed, labels), labels
                )
            distillation_loss = classification_loss.detach() * 0.0
            if teacher is not None:
                with torch.no_grad():
                    teacher_features = teacher.encode(images)
                    teacher_logits = teacher.decode(teacher_features)
                temperature = distillation_temperature
                distillation_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature**2)
            task_loss = (
                classification_loss
                + margin_weight * margin_loss
                + distillation_weight * distillation_loss
            )

            if objective is None:
                total_loss = task_loss
                privacy_loss = task_loss.detach() * 0.0
            else:
                regularizer_kwargs = {}
                if (
                    isinstance(regularizer, PrototypeCEMRegularizer)
                    and model.channel_noise is not None
                ):
                    regularizer_kwargs["noise_variance"] = (
                        model.channel_noise_variance(noise_std)
                    )
                combined = objective(
                    task_loss,
                    smashed,
                    labels,
                    **regularizer_kwargs,
                )
                total_loss = combined.total_loss
                privacy_loss = combined.privacy_loss
                batch_diagnostics = {
                    name: float(value.detach().cpu())
                    for name, value in combined.diagnostics.items()
                }

            diagnostic_batch = (
                gradient_diagnostic_interval > 0
                and batch_index % gradient_diagnostic_interval == 0
            )
            compose_gradient = (
                objective is not None
                and gradient_composition == "task_priority"
                and effective_privacy_weight > 0
                and privacy_loss.requires_grad
            )
            task_gradient = None
            privacy_gradient = None
            if diagnostic_batch or compose_gradient:
                task_gradient = torch.autograd.grad(
                    task_loss, smashed, retain_graph=True
                )[0]
                if privacy_loss.requires_grad:
                    privacy_gradient = torch.autograd.grad(
                        privacy_loss,
                        smashed,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                else:
                    privacy_gradient = None
            if diagnostic_batch and task_gradient is not None:
                gradient_task_norm_total += float(
                    task_gradient.flatten(start_dim=1).norm(dim=1).mean().detach()
                )
                if privacy_gradient is not None:
                    gradient_privacy_norm_total += float(
                        privacy_gradient.flatten(start_dim=1)
                        .norm(dim=1)
                        .mean()
                        .detach()
                    )
                gradient_diagnostic_batches += 1

            gradient_hook = None
            if compose_gradient and privacy_gradient is not None:
                composition = task_priority_gradient(
                    task_gradient,
                    privacy_gradient,
                    effective_privacy_weight,
                )
                replacement_gradient = composition.gradient.detach()
                gradient_hook = smashed.register_hook(
                    lambda _incoming, replacement=replacement_gradient: replacement
                )
                gradient_conflict_fraction_total += float(
                    composition.conflict_fraction.detach()
                )
                gradient_cosine_total += float(composition.mean_cosine.detach())
                gradient_composition_batches += 1

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if gradient_hook is not None:
                gradient_hook.remove()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

            batch_size = images.shape[0]
            totals["classification"] += float(classification_loss.detach()) * batch_size
            totals["margin"] += float(margin_loss.detach()) * batch_size
            totals["distillation"] += float(distillation_loss.detach()) * batch_size
            totals["task"] += float(task_loss.detach()) * batch_size
            totals["privacy"] += float(privacy_loss.detach()) * batch_size
            totals["total"] += float(total_loss.detach()) * batch_size
            totals["count"] += batch_size
            if objective is not None:
                for name, value in batch_diagnostics.items():
                    if name in {"initialized_prototypes", "prototype_memory_bytes"}:
                        diagnostic_last[name] = value
                    else:
                        diagnostic_totals[name] = (
                            diagnostic_totals.get(name, 0.0) + value * batch_size
                        )

        scheduler.step()
        training_elapsed = time.perf_counter() - training_started
        validation = evaluate(
            model,
            loaders.validation,
            device,
            noise_std,
            evaluation_seed=seed + 100_000,
        )
        if isinstance(regularizer, OfficialCEMRegularizer):
            refresh_record = refresh_class_statistics(
                model,
                loaders.statistics,
                regularizer,
                loaders.num_classes,
                device,
            )
        diagnostic_means = {
            name: value / totals["count"]
            for name, value in diagnostic_totals.items()
        }
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        projection_sha256 = (
            tensor_sha256(projector.matrix) if projector is not None else None
        )
        if projector is not None:
            projection_matrix = projector.matrix.detach()
            singular_values = torch.linalg.svdvals(projection_matrix)
            projection_diagnostics = {
                "assignment_projection_frobenius_norm": float(
                    projection_matrix.norm().cpu()
                ),
                "assignment_projection_singular_min": float(
                    singular_values.min().cpu()
                ),
                "assignment_projection_singular_max": float(
                    singular_values.max().cpu()
                ),
            }
        else:
            projection_diagnostics = {}
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "base_privacy_weight": base_privacy_weight,
            "effective_privacy_weight": effective_privacy_weight,
            "gradient_composition": gradient_composition,
            "optimizer": optimizer_name,
            "trainable_model_parts": trainable_model_parts,
            "margin_weight": margin_weight,
            "distillation_weight": distillation_weight,
            "distillation_temperature": distillation_temperature,
            "distillation_teacher_checkpoint": teacher_checkpoint_path,
            "train_classification_loss": totals["classification"] / totals["count"],
            "train_margin_loss": totals["margin"] / totals["count"],
            "train_distillation_loss": totals["distillation"] / totals["count"],
            "train_task_loss": totals["task"] / totals["count"],
            "train_privacy_loss": totals["privacy"] / totals["count"],
            "train_total_loss": totals["total"] / totals["count"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "training_epoch_seconds": training_elapsed,
            "total_epoch_seconds": time.perf_counter() - total_epoch_started,
            "peak_gpu_memory_bytes": peak_memory,
            "smashed_feature_elements": feature_dim,
            "transmitted_bytes_per_sample_fp32": feature_dim * 4,
            "assignment_projection_sha256": projection_sha256,
            **projection_diagnostics,
            "gradient_diagnostic_batches": gradient_diagnostic_batches,
            "smashed_task_gradient_norm": (
                gradient_task_norm_total / gradient_diagnostic_batches
                if gradient_diagnostic_batches
                else 0.0
            ),
            "smashed_privacy_gradient_norm": (
                gradient_privacy_norm_total / gradient_diagnostic_batches
                if gradient_diagnostic_batches
                else 0.0
            ),
            "privacy_task_conflict_fraction": (
                gradient_conflict_fraction_total / gradient_composition_batches
                if gradient_composition_batches
                else 0.0
            ),
            "privacy_task_gradient_cosine": (
                gradient_cosine_total / gradient_composition_batches
                if gradient_composition_batches
                else 0.0
            ),
            **refresh_record,
            **diagnostic_means,
            **diagnostic_last,
            **initialization_record,
        }
        if model.channel_noise is not None:
            multipliers = model.channel_noise.channel_multipliers().detach().cpu()
            record.update(
                {
                    "noise_channel_multiplier_min": float(multipliers.min()),
                    "noise_channel_multiplier_max": float(multipliers.max()),
                    "noise_channel_multiplier_rms": float(
                        multipliers.square().mean().sqrt()
                    ),
                    "noise_channel_multipliers": multipliers.tolist(),
                }
            )
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "regularizer_state": regularizer.state_dict() if regularizer else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "class_to_index": loaders.class_to_index,
            "smashed_shape": smashed_shape,
            "config": config,
            "validation": validation,
            "assignment_projection_sha256": projection_sha256,
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")

    best_checkpoint = torch.load(
        output_dir / "checkpoint_best.pt", map_location=device, weights_only=False
    )
    selection_metrics = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": float(best_checkpoint["validation"]["loss"]),
        "best_validation_accuracy": float(
            best_checkpoint["validation"]["accuracy"]
        ),
        "selection_split": "target_validation",
        "evaluation_noise_seed": seed + 100_000,
        "target_test_accessed": False,
    }
    save_json(output_dir / "target_selection_metrics.json", selection_metrics)
    if args.selection_only:
        print(json.dumps(selection_metrics, sort_keys=True))
        return

    model.load_state_dict(best_checkpoint["model_state"])
    target_test = evaluate(
        model,
        loaders.test,
        device,
        noise_std,
        evaluation_seed=seed + 200_000,
    )
    target_test_metrics = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "selection_split": "target_validation",
        "evaluation_split": "target_test",
        "evaluation_noise_seed": seed + 200_000,
        "target_test_accessed": True,
        "test_loss": target_test["loss"],
        "test_accuracy": target_test["accuracy"],
    }
    save_json(output_dir / "target_test_metrics.json", target_test_metrics)
    print(json.dumps(target_test_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
