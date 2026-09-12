#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.dual_path import (  # noqa: E402
    CalibratedLogitFusion,
    GlobalSemanticBottleneck,
)
from publication_cem.semantic_backbones import (  # noqa: E402
    build_semantic_backbone,
    checkpoint_backbone_name,
)
from scripts.train_dual_path_facescrub import (  # noqa: E402
    build_loaders,
    load_legacy,
    normalise_facescrub,
    restore_legacy_server,
    save_json,
)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def state_bytes(module: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_ms(values: list[float]) -> float:
    return statistics.mean(values) * 1000.0


@torch.no_grad()
def profile(
    legacy,
    semantic,
    fusion,
    loader,
    device: torch.device,
    warmup: int,
    batches: int,
) -> dict:
    client_times: list[float] = []
    server_times: list[float] = []
    total_times: list[float] = []
    spatial_shape = None
    token_shape = None
    samples = 0
    for index, (images, _labels) in enumerate(loader):
        if index >= warmup + batches:
            break
        images = images.to(device, non_blocking=True)
        synchronise(device)
        total_started = time.perf_counter()
        client_started = total_started
        spatial = legacy.local(normalise_facescrub(images))
        token = semantic.encode(images)
        synchronise(device)
        client_elapsed = time.perf_counter() - client_started

        server_started = time.perf_counter()
        hidden = legacy.cloud(spatial)
        hidden = F.adaptive_avg_pool2d(hidden, output_size=1).flatten(start_dim=1)
        legacy_output = legacy.classifier(hidden)
        semantic_output = semantic.classifier(token)
        fusion(legacy_output, semantic_output)
        synchronise(device)
        server_elapsed = time.perf_counter() - server_started
        total_elapsed = time.perf_counter() - total_started
        spatial_shape = tuple(int(value) for value in spatial.shape[1:])
        token_shape = tuple(int(value) for value in token.shape[1:])
        if index >= warmup:
            batch_size = images.shape[0]
            samples += batch_size
            client_times.append(client_elapsed / batch_size)
            server_times.append(server_elapsed / batch_size)
            total_times.append(total_elapsed / batch_size)
    if not client_times or spatial_shape is None or token_shape is None:
        raise ValueError("not enough profiling batches")
    spatial_elements = int(torch.tensor(spatial_shape).prod())
    token_elements = int(torch.tensor(token_shape).prod())
    return {
        "profiled_batches": len(client_times),
        "profiled_samples": samples,
        "client_latency_ms_per_sample_mean": mean_ms(client_times),
        "client_latency_ms_per_sample_std": (
            statistics.stdev(client_times) * 1000.0
            if len(client_times) > 1
            else None
        ),
        "server_latency_ms_per_sample_mean": mean_ms(server_times),
        "end_to_end_latency_ms_per_sample_mean": mean_ms(total_times),
        "spatial_payload_shape": spatial_shape,
        "semantic_payload_shape": token_shape,
        "spatial_payload_elements": spatial_elements,
        "semantic_payload_elements": token_elements,
        "total_payload_elements": spatial_elements + token_elements,
        "transmitted_bytes_per_sample_fp32": 4 * (spatial_elements + token_elements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--profile-batches", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    _, loader, class_to_index = build_loaders(
        args.data_root, args.batch_size, args.workers, int(checkpoint_args["seed"])
    )
    if class_to_index != checkpoint["class_to_index"]:
        raise ValueError("checkpoint and validation class mappings differ")
    legacy = load_legacy(Path(checkpoint["legacy_checkpoint_dir"]), device)
    restore_legacy_server(legacy, checkpoint)
    backbone, backbone_channels = build_semantic_backbone(
        checkpoint_backbone_name(checkpoint_args), pretrained=False
    )
    semantic = GlobalSemanticBottleneck(
        backbone,
        backbone_channels=backbone_channels,
        token_dim=int(checkpoint_args["token_dim"]),
        num_classes=530,
        dropout=0.1,
    ).to(device)
    semantic.load_state_dict(checkpoint["semantic_state"])
    fusion = CalibratedLogitFusion().to(device)
    fusion.load_state_dict(checkpoint["fusion_state"])
    legacy.eval().requires_grad_(False)
    semantic.eval().requires_grad_(False)
    fusion.eval().requires_grad_(False)

    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "target_seed": int(checkpoint_args["seed"]),
        "semantic_backbone": checkpoint_backbone_name(checkpoint_args),
        "client_parameters": parameter_count(legacy.local)
        + parameter_count(semantic.backbone)
        + parameter_count(semantic.project),
        "server_parameters": parameter_count(legacy.cloud)
        + parameter_count(legacy.classifier)
        + parameter_count(semantic.classifier)
        + parameter_count(fusion),
        "total_model_state_bytes": state_bytes(legacy)
        + state_bytes(semantic)
        + state_bytes(fusion),
        "checkpoint_file_bytes": args.checkpoint.stat().st_size,
        **profile(
            legacy,
            semantic,
            fusion,
            loader,
            device,
            args.warmup_batches,
            args.profile_batches,
        ),
    }
    save_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
