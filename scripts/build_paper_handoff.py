#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_AUDIT_CATEGORIES = ("engineering", "protocol", "preflight", "evidence")
INCLUDED_ROOTS = (
    "configs",
    "docs",
    "environment",
    "evidence",
    "generated_runbooks",
    "paper",
    "results",
    "run_logs",
    "run_state",
    "scripts",
    "src",
    "tests",
)
TOP_LEVEL_FILES = ("README.md", "manifest.json", "pyproject.toml")
RESULT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_experiment_evidence(root: Path) -> tuple[dict, dict]:
    audit_path = root / "results/readiness_audit.json"
    statistics_path = root / "results/headline_statistical_analysis.json"
    if not audit_path.is_file() or not statistics_path.is_file():
        raise RuntimeError("final readiness audit and statistical analysis are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    category_status = audit.get("category_status", {})
    failed = [
        category
        for category in REQUIRED_AUDIT_CATEGORIES
        if category_status.get(category) != "PASS"
    ]
    if failed:
        failed_checks = [
            check["identifier"]
            for check in audit.get("checks", [])
            if check.get("category") in failed and check.get("status") != "PASS"
        ]
        raise RuntimeError(
            f"experiment evidence is incomplete; categories={failed}, checks={failed_checks}"
        )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if statistics.get("status") != "ANALYSED" or statistics.get(
        "global_h1_status"
    ) not in {"SUPPORTED", "NOT_SUPPORTED"}:
        raise RuntimeError("headline statistical decision is missing or invalid")
    return audit, statistics


def collect_files(root: Path) -> list[Path]:
    files = []
    for name in TOP_LEVEL_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    for name in INCLUDED_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            if relative.parts[0] == "results" and path.suffix.lower() not in RESULT_SUFFIXES:
                continue
            if relative.parts[0] == "evidence" and path.suffix.lower() in {".pt", ".pth"}:
                continue
            if relative.parts[0] == "run_logs" and path.suffix.lower() != ".log":
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build_bundle(root: Path, output: Path, audit: dict, statistics: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory_path = output.parent / "paper_writing_inventory.json"
    readme_path = output.parent / "交付说明.md"
    files = [
        path
        for path in collect_files(root)
        if path.resolve() not in {output.resolve(), inventory_path.resolve(), readme_path.resolve()}
    ]
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    inventory = {
        "schema_version": 1,
        "status": "EXPERIMENT_EVIDENCE_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_h1_status": statistics["global_h1_status"],
        "audit_category_status": audit["category_status"],
        "file_count": len(records),
        "files": records,
        "excluded_from_lightweight_bundle": [
            "model/decoder/discriminator checkpoints (*.pt, *.pth)",
            "downloaded dataset archives and images",
        ],
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8"
    )
    readme_path.write_text(
        "# 论文写作交付包\n\n"
        "状态：`EXPERIMENT_EVIDENCE_READY`。这表示预先规定的正式实验、统计、"
        "泛化、攻击和效率证据已经通过审计，可以据此撰写论文；不表示方法一定"
        "优于基线，也不表示投稿一定接收。\n\n"
        f"主假设统计状态：`{statistics['global_h1_status']}`。论文必须按照该状态"
        "决定是否可以使用 outperform/improve 等肯定性表述。\n\n"
        "该压缩包用于论文写作，包含配置、环境、哈希、日志、逐样本指标、统计结果、"
        "自动表图和代码，但不复制体积很大的模型与攻击器 checkpoint。完整 Publication "
        "目录应保留在服务器上，至少保存到论文评审结束，以便补充审计或重算图表。\n",
        encoding="utf-8",
    )
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        archive.add(inventory_path, arcname="handoff/paper_writing_inventory.json")
        archive.add(readme_path, arcname="handoff/交付说明.md")
    bundle_hash = sha256(output)
    (output.parent / f"{output.name}.sha256").write_text(
        f"{bundle_hash}  {output.name}\n", encoding="ascii"
    )
    return {
        "status": "EXPERIMENT_EVIDENCE_READY",
        "global_h1_status": statistics["global_h1_status"],
        "bundle": str(output),
        "bundle_bytes": output.stat().st_size,
        "bundle_sha256": bundle_hash,
        "file_count": len(records) + 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "handoff/paper_writing_bundle.tar.gz").resolve()
    try:
        audit, statistics = verify_experiment_evidence(root)
        result = build_bundle(root, output, audit, statistics)
    except RuntimeError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
