#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_protocol_loaders
from publication_cem.models import build_split_model
from publication_cem.reproducibility import save_json, seed_everything


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def state_bytes(module: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size() for tensor in module.state_dict().values()
    )


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def profile_batches(model, loader, device, noise_std: float, warmup: int, batches: int) -> dict:
    edge_times = []
    cloud_times = []
    total_times = []
    samples = 0
    for index, (images, _) in enumerate(loader):
        if index >= warmup + batches:
            break
        images = images.to(device, non_blocking=True)
        synchronise(device)
        total_started = time.perf_counter()
        edge_started = total_started
        smashed = model.encode(images)
        synchronise(device)
        edge_elapsed = time.perf_counter() - edge_started
        transmitted = model.transmit(smashed, noise_std)
        synchronise(device)
        cloud_started = time.perf_counter()
        model.decode(transmitted)
        synchronise(device)
        cloud_elapsed = time.perf_counter() - cloud_started
        total_elapsed = time.perf_counter() - total_started
        if index >= warmup:
            batch_size = images.shape[0]
            samples += batch_size
            edge_times.append(edge_elapsed / batch_size)
            cloud_times.append(cloud_elapsed / batch_size)
            total_times.append(total_elapsed / batch_size)
    if not edge_times:
        raise ValueError("not enough profiling batches")
    return {
        "profiled_batches": len(edge_times),
        "profiled_samples": samples,
        "edge_latency_ms_per_sample_mean": statistics.mean(edge_times) * 1000,
        "edge_latency_ms_per_sample_std": (
            statistics.stdev(edge_times) * 1000 if len(edge_times) > 1 else None
        ),
        "cloud_latency_ms_per_sample_mean": statistics.mean(cloud_times) * 1000,
        "end_to_end_latency_ms_per_sample_mean": statistics.mean(total_times) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--profile-batches", type=int, default=100)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loaders = build_protocol_loaders(config["data"], int(config["seed"]))
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model = build_split_model(
        model_config,
        loaders.num_classes,
        int(config["data"]["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    defense = config["defense"]
    noise_std = 0.0 if defense["name"] == "none" else float(defense["noise_std"])
    latency = profile_batches(
        model,
        loaders.test,
        device,
        noise_std,
        args.warmup_batches,
        args.profile_batches,
    )
    smashed_shape = tuple(int(value) for value in checkpoint["smashed_shape"])
    smashed_elements = int(torch.tensor(smashed_shape).prod())
    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_name": model_config["name"],
        "cut_index": model_config.get("cut_index"),
        "smashed_shape": smashed_shape,
        "smashed_feature_elements": smashed_elements,
        "transmitted_bytes_per_sample_fp32": smashed_elements * 4,
        "edge_parameters": parameter_count(model.local)
        + parameter_count(model.bottleneck_encoder),
        "cloud_parameters": parameter_count(model.bottleneck_decoder)
        + parameter_count(model.cloud)
        + parameter_count(model.classifier),
        "model_state_bytes": state_bytes(model),
        "checkpoint_file_bytes": args.checkpoint.stat().st_size,
        **latency,
    }
    output = args.output or args.checkpoint.parent / "efficiency_profile.json"
    save_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
