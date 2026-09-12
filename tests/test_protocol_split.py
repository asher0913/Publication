from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from publication_cem.data import build_imagefolder_loaders
from scripts.build_protocol_split import build_manifest


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_protocol_manifest_is_disjoint_and_loadable(tmp_path):
    for split, count in (("train", 10), ("val", 3)):
        for class_index, class_name in enumerate(("a", "b")):
            for image_index in range(count):
                write_image(
                    tmp_path / split / class_name / f"{image_index}.png",
                    class_index * 100 + image_index,
                )

    manifest = build_manifest(tmp_path, 123, 0.2, 0.2)
    manifest_path = tmp_path / "split.json"
    import json

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_paths = {
        name: set(record["paths"]) for name, record in manifest["splits"].items()
    }
    for left, left_paths in split_paths.items():
        for right, right_paths in split_paths.items():
            if left < right:
                assert left_paths.isdisjoint(right_paths)

    loaders = build_imagefolder_loaders(
        str(tmp_path),
        image_size=8,
        batch_size=2,
        workers=0,
        seed=123,
        split_manifest=str(manifest_path),
    )
    assert len(loaders.train.dataset) == 12
    assert len(loaders.validation.dataset) == 4
    assert len(loaders.attacker_auxiliary_train.dataset) == 2
    assert len(loaders.attacker_auxiliary_validation.dataset) == 2
    assert len(loaders.test.dataset) == 6
    assert loaders.num_classes == 2

    nominal_loaders = build_imagefolder_loaders(
        str(tmp_path),
        image_size=8,
        batch_size=2,
        workers=0,
        seed=123,
        split_manifest=str(manifest_path),
        num_classes_override=3,
    )
    assert len(nominal_loaders.class_to_index) == 2
    assert nominal_loaders.num_classes == 3

    with pytest.raises(ValueError, match="smaller than observed"):
        build_imagefolder_loaders(
            str(tmp_path),
            image_size=8,
            batch_size=2,
            workers=0,
            seed=123,
            split_manifest=str(manifest_path),
            num_classes_override=1,
        )

    paper_loaders = build_imagefolder_loaders(
        str(tmp_path),
        image_size=8,
        batch_size=2,
        workers=0,
        seed=123,
        protocol_mode="cem_facescrub_paper",
    )
    assert paper_loaders.protocol_mode == "cem_facescrub_paper"
    assert len(paper_loaders.attacker_auxiliary_train.dataset) == 6
    assert len(paper_loaders.attacker_auxiliary_validation.dataset) == 0


def test_imagefolder_loader_rejects_a_tampered_split_hash(tmp_path):
    for split, count in (("train", 10), ("val", 3)):
        for class_index, class_name in enumerate(("a", "b")):
            for image_index in range(count):
                write_image(
                    tmp_path / split / class_name / f"{image_index}.png",
                    class_index * 100 + image_index,
                )
    manifest = build_manifest(tmp_path, 123, 0.2, 0.2)
    manifest["splits"]["target_train"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "tampered.json"
    import json

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path hash"):
        build_imagefolder_loaders(
            str(tmp_path),
            image_size=8,
            batch_size=2,
            workers=0,
            seed=123,
            split_manifest=str(manifest_path),
        )
