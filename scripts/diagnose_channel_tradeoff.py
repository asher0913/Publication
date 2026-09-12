#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_protocol_loaders
from publication_cem.models import (
    build_inversion_decoder,
    build_split_model,
)
from publication_cem.reproducibility import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attack-type", default="conv_decoder")
    parser.add_argument("--batches", type=int, default=32)
    return parser.parse_args()


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return 0.0
    return float((left @ right) / denominator)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    data_cfg = config["data"]
    loaders = build_protocol_loaders(data_cfg, seed)
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    model = build_split_model(
        model_cfg,
        num_classes=loaders.num_classes,
        image_size=int(data_cfg["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)

    attack_cfg = config["attack"]
    decoder = build_inversion_decoder(
        args.attack_type,
        input_channels=int(checkpoint["smashed_shape"][0]),
        output_size=int(data_cfg["image_size"]),
        width=int(attack_cfg.get("decoder_width", 128)),
    ).to(device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=device, weights_only=True))
    decoder.eval().requires_grad_(False)

    defense_cfg = config["defense"]
    noise_std = (
        0.0 if defense_cfg["name"] == "none" else float(defense_cfg["noise_std"])
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 700_000)

    task_sensitivity = None
    reconstruction_sensitivity = None
    activation_variance = None
    example_count = 0
    batch_count = 0
    for batch_index, (images, labels) in enumerate(loaders.attacker_auxiliary_validation):
        if batch_index >= args.batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            clean = model.encode(images)
        smashed = clean.detach().requires_grad_(True)
        transmitted = model.transmit(smashed, noise_std, generator)

        task_loss = F.cross_entropy(model.decode(transmitted), labels)
        task_gradient = torch.autograd.grad(task_loss, smashed, retain_graph=True)[0]
        reconstruction_loss = F.mse_loss(decoder(transmitted), images)
        reconstruction_gradient = torch.autograd.grad(reconstruction_loss, smashed)[0]

        reduce_dims = (0, *range(2, smashed.ndim))
        batch_size = images.shape[0]
        task_batch = task_gradient.square().mean(dim=reduce_dims).detach().cpu()
        reconstruction_batch = (
            reconstruction_gradient.square().mean(dim=reduce_dims).detach().cpu()
        )
        activation_batch = clean.var(dim=reduce_dims, unbiased=False).detach().cpu()
        if task_sensitivity is None:
            task_sensitivity = torch.zeros_like(task_batch)
            reconstruction_sensitivity = torch.zeros_like(reconstruction_batch)
            activation_variance = torch.zeros_like(activation_batch)
        task_sensitivity += task_batch * batch_size
        reconstruction_sensitivity += reconstruction_batch * batch_size
        activation_variance += activation_batch * batch_size
        example_count += batch_size
        batch_count += 1

    if not example_count:
        raise ValueError("diagnostic loader produced no examples")
    task_sensitivity /= example_count
    reconstruction_sensitivity /= example_count
    activation_variance /= example_count

    task_share = task_sensitivity / task_sensitivity.sum().clamp_min(1e-20)
    reconstruction_share = reconstruction_sensitivity / reconstruction_sensitivity.sum().clamp_min(
        1e-20
    )
    privacy_to_task_ratio = reconstruction_share / task_share.clamp_min(1e-12)
    channel_count = task_share.numel()
    top_count = max(1, channel_count // 4)
    task_top = set(torch.topk(task_share, top_count).indices.tolist())
    reconstruction_top = set(
        torch.topk(reconstruction_share, top_count).indices.tolist()
    )
    opportunity_top = torch.topk(privacy_to_task_ratio, top_count).indices.tolist()

    channels = []
    for index in range(channel_count):
        channels.append(
            {
                "channel": index,
                "task_gradient_share": float(task_share[index]),
                "reconstruction_gradient_share": float(reconstruction_share[index]),
                "reconstruction_to_task_ratio": float(privacy_to_task_ratio[index]),
                "activation_variance": float(activation_variance[index]),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "decoder": str(args.decoder),
        "attack_type": args.attack_type,
        "noise_std": noise_std,
        "batches": batch_count,
        "examples": example_count,
        "channels": channels,
        "pearson_task_reconstruction": correlation(
            task_sensitivity, reconstruction_sensitivity
        ),
        "spearman_task_reconstruction": correlation(
            rankdata(task_sensitivity), rankdata(reconstruction_sensitivity)
        ),
        "top_quartile_overlap": len(task_top & reconstruction_top) / top_count,
        "task_top_channels": sorted(task_top),
        "reconstruction_top_channels": sorted(reconstruction_top),
        "high_reconstruction_low_task_channels": opportunity_top,
        "max_reconstruction_to_task_ratio": float(privacy_to_task_ratio.max()),
        "median_reconstruction_to_task_ratio": float(privacy_to_task_ratio.median()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
