#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "manifest.json").resolve()
    excluded_roots = {root / "results", root / ".git", root / ".pytest_cache"}
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        if any(excluded in path.resolve().parents for excluded in excluded_roots):
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.name,
        "file_count": len(records),
        "files": records,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
