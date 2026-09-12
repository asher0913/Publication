#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCUMENTS = {
    "UA-CEM正式实验说明.md",
    "中文论文发表实施与材料交付报告.md",
    "导师说明_项目内容与论文发表可行性.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "handoff/uacem_paper_writing_bundle.tar.gz",
    )
    args = parser.parse_args()
    audit_path = ROOT / "results/uacem_formal/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("UA-CEM evidence audit has not passed")
    roots = (
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "environment",
        ROOT / "evidence",
        ROOT / "generated_runbooks/uacem_formal",
        ROOT / "paper",
        ROOT / "paper/generated/uacem",
        ROOT / "results/uacem_formal",
        ROOT / "run_logs/uacem_formal",
        ROOT / "run_state/uacem_formal",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
    )
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.parent == ROOT / "docs" and path.name not in CURRENT_DOCUMENTS:
                continue
            if path.suffix.lower() in {
                ".pt", ".pth", ".aux", ".bbl", ".blg", ".fdb_latexmk",
                ".fls", ".log", ".out",
            }:
                continue
            files.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    payload = {
        "status": "EXPERIMENT_EVIDENCE_READY",
        "bundle": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "files": len(set(files)),
    }
    (args.output.parent / "uacem_paper_writing_bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
