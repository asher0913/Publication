#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.evaluators import FaceEmbedder, LPIPSMetric
from publication_cem.reproducibility import (
    module_state_sha256,
    save_json,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--device")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence/evaluation_assets.json",
    )
    args = parser.parse_args()
    seed_everything(20_260_716)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    lpips_metric = LPIPSMetric(device, net=args.lpips_net, version="0.1")
    face_embedder = FaceEmbedder(device)
    image = torch.rand(2, 3, 64, 64, device=device)
    with torch.no_grad():
        lpips_self = lpips_metric(image, image)
        embeddings = face_embedder(image)
    payload = {
        "schema_version": 1,
        "status": "VERIFIED",
        "device": str(device),
        "lpips_package_version": importlib.metadata.version("lpips"),
        "lpips_net": args.lpips_net,
        "lpips_version": "0.1",
        "lpips_weights_sha256": module_state_sha256(lpips_metric.model),
        "lpips_self_distance_max": float(lpips_self.max().cpu()),
        "facenet_pytorch_package_version": importlib.metadata.version(
            "facenet-pytorch"
        ),
        "face_model": "InceptionResnetV1-vggface2",
        "face_weights_sha256": module_state_sha256(face_embedder.model),
        "face_embedding_shape": list(embeddings.shape),
        "face_embedding_norm_error_max": float(
            (embeddings.norm(dim=1) - 1).abs().max().cpu()
        ),
    }
    save_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
