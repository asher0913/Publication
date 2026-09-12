#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.dual_path_attack import DualPathReconstructor  # noqa: E402
from scripts.attack_dual_path_facescrub import (  # noqa: E402
    build_attack_loaders,
    load_feature_extractors,
    transmitted_features,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reconstructor(path: Path, semantic_dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = DualPathReconstructor(
        semantic_dim=semantic_dim,
        width=int(args["width"]),
        residual_blocks=int(args["residual_blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["reconstructor_state"])
    return model.eval().requires_grad_(False)


def render(
    rows: list[tuple[str, torch.Tensor]],
    output: Path,
    scale: int = 4,
) -> None:
    if scale <= 0:
        raise ValueError("output scale must be positive")
    tile_width, tile_height = to_pil_image(rows[0][1][0]).size
    tile_width *= scale
    tile_height *= scale
    label_width = 120 * scale
    padding = 5 * scale
    canvas = Image.new(
        "RGB",
        (
            label_width + len(rows[0][1]) * (tile_width + padding) + padding,
            len(rows) * (tile_height + padding) + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13 * scale)
    except OSError:
        font = ImageFont.load_default()
    for row_index, (label, tensors) in enumerate(rows):
        y = padding + row_index * (tile_height + padding)
        draw.text(
            (8 * scale, y + tile_height // 2),
            label,
            fill="black",
            font=font,
            anchor="lm",
        )
        for column, tensor in enumerate(tensors):
            image = to_pil_image(tensor.clamp(0, 1))
            image = image.resize(
                (tile_width, tile_height),
                resample=Image.Resampling.LANCZOS,
            )
            x = label_width + column * (tile_width + padding)
            canvas.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--gan-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--legacy-noise-std", type=float, default=0.31)
    parser.add_argument("--semantic-noise-std", type=float, default=0.10)
    parser.add_argument("--noise-seed", type=int, default=500125)
    parser.add_argument("--output-scale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("sample count must be positive")

    device = torch.device(args.device)
    _, evaluation_loader = build_attack_loaders(
        args.data_root,
        batch_size=args.samples,
        workers=0,
        attack_knowledge="inference",
    )
    images, labels = next(iter(evaluation_loader))
    images = images[: args.samples].to(device)
    labels = labels[: args.samples]
    legacy, semantic, _legacy_noise, _semantic_noise, token_dim = (
        load_feature_extractors(args.target_checkpoint, device)
    )
    decoder = load_reconstructor(args.decoder_checkpoint, token_dim, device)
    gan = load_reconstructor(args.gan_checkpoint, token_dim, device)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)
    spatial, token = transmitted_features(
        legacy,
        semantic,
        images,
        args.legacy_noise_std,
        args.semantic_noise_std,
        generator,
    )
    rows = [
        ("Original", images.cpu()),
        ("Decoder", decoder(spatial, token).cpu()),
        ("GAN", gan(spatial, token).cpu()),
    ]
    render(rows, args.output, args.output_scale)
    provenance = {
        "selection_rule": "first samples in the fixed inference-knowledge evaluation split",
        "labels": [int(label) for label in labels],
        "noise_seed": args.noise_seed,
        "effective_legacy_noise_std": args.legacy_noise_std,
        "effective_semantic_noise_std": args.semantic_noise_std,
        "output_scale": args.output_scale,
        "target_checkpoint": str(args.target_checkpoint),
        "target_checkpoint_sha256": file_sha256(args.target_checkpoint),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "decoder_checkpoint_sha256": file_sha256(args.decoder_checkpoint),
        "gan_checkpoint": str(args.gan_checkpoint),
        "gan_checkpoint_sha256": file_sha256(args.gan_checkpoint),
        "output_sha256": file_sha256(args.output),
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "samples": args.samples}))


if __name__ == "__main__":
    main()
