#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("supplementary aggregation has not passed")
    subprocess.run(
        ["bash", "scripts/finalize_pattern_recognition_paper.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "PUBLICATION_ROOT": str(ROOT),
            "PUBLICATION_PYTHON": os.environ.get("PUBLICATION_PYTHON", os.sys.executable),
        },
        check=True,
    )
    finalisation_path = (
        ROOT / "results" / "dual_path_publication" / "paper_finalization.json"
    )
    finalisation = json.loads(finalisation_path.read_text(encoding="utf-8"))
    payload = {
        "status": "PASS",
        "supplementary_summary": str(args.summary),
        "supplementary_summary_sha256": hashlib.sha256(
            args.summary.read_bytes()
        ).hexdigest(),
        "paper_finalization": str(finalisation_path),
        **finalisation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
