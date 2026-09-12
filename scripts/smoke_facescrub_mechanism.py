#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.config import PrototypeCEMConfig
from publication_cem.data import build_protocol_loaders
from publication_cem.metrics import reconstruction_metrics
from publication_cem.models import (
    build_inversion_decoder,
    build_split_model,
)
from publication_cem.objective import PrivacyUtilityObjective
from publication_cem.regularizer import PrototypeCEMRegularizer
from publication_cem.reproducibility import save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/actual_facescrub_mechanism_smoke.json",
    )
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/facescrub_base.json").read_text(encoding="utf-8")
    )
    config["data"].update(batch_size=4, workers=0)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    loaders = build_protocol_loaders(config["data"], int(config["seed"]))
    model = build_split_model(
        config["model"], loaders.num_classes, int(config["data"]["image_size"])
    ).to(device)
    smashed_shape = model.smashed_shape((3, 64, 64))
    feature_dim = int(torch.tensor(smashed_shape).prod())
    regularizer_config = dict(config["defense"]["regularizer"])
    regularizer_config.update(
        feature_dim=feature_dim,
        num_classes=loaders.num_classes,
        noise_std=float(config["defense"]["noise_std"]),
        seed=int(config["seed"]),
    )
    regularizer = PrototypeCEMRegularizer(
        PrototypeCEMConfig(**regularizer_config)
    ).to(device)
    objective = PrivacyUtilityObjective(
        regularizer, float(config["defense"]["privacy_weight"])
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    steps = []
    last_images = None
    last_smashed = None
    for step, (images, labels) in enumerate(loaders.train, start=1):
        images = images.to(device)
        labels = labels.to(device)
        smashed = model.encode(images)
        transmitted = model.transmit(
            smashed, float(config["defense"]["noise_std"])
        )
        task_loss = F.cross_entropy(model.decode(transmitted), labels)
        combined = objective(task_loss, smashed, labels)
        optimizer.zero_grad(set_to_none=True)
        combined.total_loss.backward()
        encoder_gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().square().sum())
                for parameter in model.local.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()
        steps.append(
            {
                "step": step,
                "labels": [int(label) for label in labels.cpu()],
                "task_loss": float(task_loss.detach()),
                "privacy_loss": float(combined.privacy_loss.detach()),
                "total_finite": bool(torch.isfinite(combined.total_loss)),
                "encoder_grad_norm": encoder_gradient_norm,
                "initialized_prototypes": int(regularizer.bank.valid.sum()),
            }
        )
        last_images = images.detach()
        last_smashed = smashed.detach()
        if step == 3:
            break
    if last_images is None or last_smashed is None:
        raise ValueError("FaceScrub target-train split is empty")
    decoder = build_inversion_decoder(
        "residual_decoder",
        input_channels=smashed_shape[0],
        output_size=64,
        width=16,
    ).to(device)
    reconstruction = decoder(last_smashed)
    decoder_loss = F.mse_loss(reconstruction, last_images)
    decoder_loss.backward()
    metrics = reconstruction_metrics(reconstruction.detach(), last_images)
    attack_record = {
        "reconstruction_shape": list(reconstruction.shape),
        "loss_finite": bool(torch.isfinite(decoder_loss)),
        **{name: float(value) for name, value in metrics.items()},
    }
    passed = (
        loaders.num_classes == 526
        and all(step["total_finite"] and step["encoder_grad_norm"] > 0 for step in steps)
        and attack_record["loss_finite"]
        and all(math.isfinite(attack_record[name]) for name in ("mse", "ssim", "psnr"))
    )
    payload = {
        "schema_version": 2,
        "status": "PASS" if passed else "FAIL",
        "purpose": "Real-data mechanism smoke only; not a paper result.",
        "data": {
            "root": config["data"]["root"],
            "split_manifest": config["data"]["split_manifest"],
            "num_classes": loaders.num_classes,
            "batch_size": 4,
            "image_shape": list(last_images.shape),
        },
        "model": {
            **config["model"],
            "smashed_shape": [4, *smashed_shape],
            "logit_shape": [4, loaders.num_classes],
        },
        "steps": steps,
        "attack_step": attack_record,
    }
    save_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
