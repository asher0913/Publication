#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.data import build_protocol_loaders  # noqa: E402
from publication_cem.models import (  # noqa: E402
    build_inversion_decoder,
    build_split_model,
)
from publication_cem.reproducibility import seed_everything  # noqa: E402


METHODS = (
    ("no_defense", "No defence"),
    ("gaussian", "Gaussian"),
    ("official_cem", "Official CEM"),
    ("single_prototype", "Single prototype"),
    ("slots8", "ProtoSlot-CEM"),
)
UACEM_METHODS = (
    ("no_defense", "No defence"),
    ("gaussian", "Gaussian"),
    ("official_cem", "Official CEM"),
    ("uacem", "UA-CEM"),
)
ATTACKER_SEEDS = (10125, 20125, 30125)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative_runs(
    results_root: Path,
    attack_seed: int,
    methods: tuple[tuple[str, str], ...] = METHODS,
) -> dict[str, dict]:
    selected = {}
    for suffix, label in methods:
        candidates = []
        for run_dir in sorted(results_root.glob(f"seed12[5-9]_{suffix}")):
            metrics_paths = [
                run_dir
                / "attacks"
                / "conv_decoder"
                / f"seed{seed}"
                / "attack_metrics.json"
                for seed in ATTACKER_SEEDS
            ]
            if not all(path.is_file() for path in metrics_paths):
                continue
            mse_values = [
                float(
                    json.loads(path.read_text(encoding="utf-8")).get(
                        "mse", math.nan
                    )
                )
                for path in metrics_paths
            ]
            if all(math.isfinite(value) for value in mse_values):
                display_path = (
                    run_dir
                    / "attacks"
                    / "conv_decoder"
                    / f"seed{attack_seed}"
                    / "attack_metrics.json"
                )
                candidates.append((run_dir, display_path, mean(mse_values)))
        if len(candidates) != 5:
            raise ValueError(f"{suffix} requires five complete target seeds")
        average = mean(value for _, _, value in candidates)
        run_dir, metrics_path, mse = min(
            candidates, key=lambda row: abs(row[2] - average)
        )
        selected[suffix] = {
            "label": label,
            "run_dir": run_dir,
            "metrics_path": metrics_path,
            "mse": mse,
            "group_mean_mse": average,
        }
    return selected


def first_batch(checkpoint: dict, samples: int):
    config = checkpoint["config"]
    loaders = build_protocol_loaders(config["data"], int(config["seed"]))
    images = []
    labels = []
    for batch_images, batch_labels in loaders.test:
        take = min(samples - len(labels), batch_images.shape[0])
        images.append(batch_images[:take])
        labels.extend(int(value) for value in batch_labels[:take])
        if len(labels) == samples:
            break
    if len(labels) != samples:
        raise ValueError("target-test split does not contain enough samples")
    sample_ids = None
    manifest = loaders.split_manifest
    if manifest is not None:
        record = manifest["splits"]["target_test"]
        values = record.get("paths") or [f"test:{index}" for index in record["indices"]]
        sample_ids = values[:samples]
    return torch.cat(images), labels, sample_ids


@torch.no_grad()
def reconstruct(run: dict, images: torch.Tensor, device: torch.device, attack_seed: int):
    checkpoint_path = run["run_dir"] / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model = build_split_model(
        model_config,
        len(checkpoint["class_to_index"]),
        int(config["data"]["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    attack_cfg = config["attack"]
    decoder = build_inversion_decoder(
        "conv_decoder",
        int(checkpoint["smashed_shape"][0]),
        int(config["data"]["image_size"]),
        int(attack_cfg["decoder_width"]),
    ).to(device)
    decoder_path = run["metrics_path"].parent / "decoder.pt"
    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True)
    )
    decoder.eval().requires_grad_(False)
    seed_everything(attack_seed)
    evaluation_noise_seed = int(config["seed"]) + 300_000
    generator = torch.Generator(device=device)
    generator.manual_seed(evaluation_noise_seed)
    clean = model.encode(images.to(device))
    defense = config["defense"]
    noise_std = 0.0 if defense["name"] == "none" else float(defense["noise_std"])
    transmitted = model.transmit(clean, noise_std, generator)
    output = decoder(transmitted).cpu()
    return output, checkpoint_path, decoder_path


def render(rows: list[tuple[str, torch.Tensor]], output: Path) -> None:
    sample_image = to_pil_image(rows[0][1][0].clamp(0, 1))
    tile_width, tile_height = sample_image.size
    label_width = 150
    padding = 4
    canvas = Image.new(
        "RGB",
        (
            label_width + len(rows[0][1]) * (tile_width + padding) + padding,
            len(rows) * (tile_height + padding) + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, tensors) in enumerate(rows):
        y = padding + row_index * (tile_height + padding)
        draw.text((8, y + tile_height // 2 - 5), label, fill="black")
        for column, tensor in enumerate(tensors):
            image = to_pil_image(tensor.clamp(0, 1))
            x = label_width + column * (tile_width + padding)
            canvas.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attack-seed", type=int, default=10125)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--method-set", choices=("legacy", "uacem"), default="legacy")
    args = parser.parse_args()
    methods = UACEM_METHODS if args.method_set == "uacem" else METHODS
    selected = representative_runs(args.results_root, args.attack_seed, methods)
    first_run = next(iter(selected.values()))
    checkpoint = torch.load(
        first_run["run_dir"] / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    originals, labels, sample_ids = first_batch(checkpoint, args.samples)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = [("Original", originals)]
    provenance_runs = {}
    for suffix, _ in methods:
        run = selected[suffix]
        reconstructions, checkpoint_path, decoder_path = reconstruct(
            run, originals, device, args.attack_seed
        )
        rows.append((run["label"], reconstructions))
        provenance_runs[suffix] = {
            "run_dir": str(run["run_dir"]),
            "mse": run["mse"],
            "group_mean_mse": run["group_mean_mse"],
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "decoder_sha256": file_sha256(decoder_path),
        }
    render(rows, args.output)
    provenance = {
        "schema_version": 1,
        "selection_rule": (
            "target seed closest to the five-target mean after averaging three "
            "conv-decoder attacker seeds within each target"
        ),
        "attacker_seed": args.attack_seed,
        "sample_rule": "first N examples in frozen target-test order",
        "sample_ids": sample_ids,
        "labels": labels,
        "runs": provenance_runs,
        "output_sha256": file_sha256(args.output),
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "samples": args.samples}))


if __name__ == "__main__":
    main()
