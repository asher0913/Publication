#!/usr/bin/env bash
set -euo pipefail

ROOT="${PUBLICATION_ROOT:-/home/unnc/zhang/Publication}"
PYTHON="${PUBLICATION_PYTHON:-/home/unnc/miniconda3/envs/publication-cem/bin/python}"
SUMMARY="$ROOT/results/dual_path_publication/formal_summary.json"
SUPPLEMENTARY_SUMMARY="$ROOT/results/dual_path_supplementary/supplementary_summary.json"
PAPER="$ROOT/paper"
SUBMISSION="$ROOT/submission/journal"
OUTPUT="$ROOT/results/dual_path_publication"
HIGHLIGHTS="$PAPER/Highlights.txt"
QUALITATIVE_GRID="$PAPER/generated/dual_path/qualitative_grid.png"
REFINED_QUALITATIVE_GRID="$PAPER/generated/dual_path/qualitative_grid_refined.png"

"$PYTHON" "$ROOT/scripts/generate_publication_figures.py" \
    --formal-summary "$SUMMARY" \
    --output-dir "$PAPER/generated/dual_path" \
    --qualitative-source "$QUALITATIVE_GRID"

"$PYTHON" - \
    "$SUMMARY" "$SUPPLEMENTARY_SUMMARY" "$HIGHLIGHTS" "$REFINED_QUALITATIVE_GRID" <<'PY'
import json
import sys
from pathlib import Path

from PIL import Image

main_path, supplementary_path, highlights_path, grid_path = map(Path, sys.argv[1:5])
if not main_path.is_file():
    raise SystemExit(f"missing formal summary: {main_path}")
if not supplementary_path.is_file():
    raise SystemExit(f"missing supplementary summary: {supplementary_path}")
main = json.loads(main_path.read_text(encoding="utf-8"))
supplementary = json.loads(supplementary_path.read_text(encoding="utf-8"))
if main.get("status") != "PASS":
    raise SystemExit("refusing to finalise a paper whose formal gate did not pass")
if supplementary.get("status") != "PASS":
    raise SystemExit("refusing to finalise before supplementary aggregation passes")
if supplementary.get("capacity_match", {}).get("status") != "PASS":
    raise SystemExit("refusing to finalise before the capacity check passes")
highlights = [
    line.strip()
    for line in highlights_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not 3 <= len(highlights) <= 5:
    raise SystemExit("Highlights must contain three to five non-empty lines")
too_long = [line for line in highlights if len(line) > 85]
if too_long:
    raise SystemExit(f"Highlight exceeds 85 characters: {too_long[0]}")
with Image.open(grid_path) as grid:
    if grid.width < 2400 or grid.height < 800:
        raise SystemExit(
            f"qualitative grid is below publication resolution: {grid.size}"
        )
PY

test -s "$PAPER/generated/dual_path_supplementary/controls_table.tex"
test -s "$PAPER/generated/dual_path_supplementary/controls_narrative.tex"

cd "$PAPER"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >main.pdflatex1.log
bibtex main >main.bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >main.pdflatex2.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >main.pdflatex3.log

if grep -Eiq "undefined (citation|citations|reference|references)|Citation .* undefined|Reference .* undefined" main.log; then
    echo "Undefined citation or reference remains in paper/main.log" >&2
    exit 3
fi

pdftotext main.pdf main.audit.txt
if grep -Eiq "formal campaign pending|numerical conclusion is withheld|formal artefacts.*incomplete" main.audit.txt; then
    echo "Unresolved experimental placeholder remains in the manuscript" >&2
    exit 4
fi
rm -f main.audit.txt

pages="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
if ! pdftotext -f "$pages" -l "$pages" -layout main.pdf - | \
    grep -Eq "Page[[:space:]]+$pages[[:space:]]+of[[:space:]]+$pages"; then
    echo "Manuscript footer page count does not match the PDF page count" >&2
    exit 5
fi
if pdffonts main.pdf | awk 'NR > 2 && $2 == "Type" && $3 == "3" {found=1} END {exit !found}'; then
    echo "Type 3 font remains in the manuscript" >&2
    exit 6
fi

mkdir -p "$OUTPUT"
cp main.pdf "$OUTPUT/Pattern_Recognition_DualPath_CEM.pdf"

cd "$SUBMISSION"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory=build Cover_Letter_Pattern_Recognition.tex \
    >build/Cover_Letter_Pattern_Recognition.pdflatex.log
cp build/Cover_Letter_Pattern_Recognition.pdf \
    "$OUTPUT/Cover_Letter_Pattern_Recognition.pdf"

cd "$PAPER"
tar -czf "$OUTPUT/Pattern_Recognition_DualPath_CEM_source.tar.gz" \
    main.tex references.bib cas-sc.cls cas-common.sty cas-model2-names.bst \
    sections thumbnails generated/dual_path generated/dual_path_supplementary \
    highlights.tex Highlights.txt

"$PYTHON" - "$OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
checkpoints = []
for path in sorted(output.rglob("checkpoint_best.pt")):
    checkpoints.append(
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
manifest = output / "checkpoint_manifest.json"
manifest.write_text(
    json.dumps({"schema_version": 1, "checkpoints": checkpoints}, indent=2) + "\n"
)
PY

rm -f "$OUTPUT/paper_finalization.json" \
    "$OUTPUT/Pattern_Recognition_DualPath_CEM_evidence.tar.gz"
evidence_tmp="$(mktemp /tmp/dual_path_evidence.XXXXXX)"
tar -czf "$evidence_tmp" \
    --exclude='*.pt' \
    --exclude='Pattern_Recognition_DualPath_CEM_evidence.tar.gz' \
    -C "$ROOT" \
    results/dual_path_publication \
    results/dual_path_supplementary \
    run_state/dual_path_publication.json \
    run_state/dual_path_supplementary.json \
    run_logs/dual_path_publication \
    run_logs/dual_path_supplementary \
    run_logs/dual_path_publication_master_20260804.log
mv "$evidence_tmp" "$OUTPUT/Pattern_Recognition_DualPath_CEM_evidence.tar.gz"

package_files=(
    results/dual_path_publication/Pattern_Recognition_DualPath_CEM.pdf
    results/dual_path_publication/Pattern_Recognition_DualPath_CEM_source.tar.gz
    results/dual_path_publication/Cover_Letter_Pattern_Recognition.pdf
    paper/Highlights.txt
    submission/journal/Cover_Letter_Pattern_Recognition.tex
    submission/journal/State_of_the_Art_Response.md
    submission/journal/Author_Metadata.md
    submission/journal/Submission_Checklist.md
)
for optional in \
    submission/journal/Declaration_of_Interest.txt \
    submission/journal/AI_Use_Declaration.txt; do
    if [[ -f "$ROOT/$optional" ]]; then
        package_files+=("$optional")
    fi
done
tar -czf "$OUTPUT/Pattern_Recognition_Submission_Package.tar.gz" \
    -C "$ROOT" "${package_files[@]}"

"$PYTHON" - "$OUTPUT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
pdf = output / "Pattern_Recognition_DualPath_CEM.pdf"
source = output / "Pattern_Recognition_DualPath_CEM_source.tar.gz"
evidence = output / "Pattern_Recognition_DualPath_CEM_evidence.tar.gz"
checkpoint_manifest = output / "checkpoint_manifest.json"
cover_letter = output / "Cover_Letter_Pattern_Recognition.pdf"
submission_package = output / "Pattern_Recognition_Submission_Package.tar.gz"
info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
pages = next(
    int(line.split(":", 1)[1].strip())
    for line in info.splitlines()
    if line.startswith("Pages:")
)
payload = {
    "status": "PASS",
    "finalised_at_utc": datetime.now(timezone.utc).isoformat(),
    "pages": pages,
    "pdf": str(pdf),
    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "source_archive": str(source),
    "source_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "evidence_archive": str(evidence),
    "evidence_archive_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    "checkpoint_manifest": str(checkpoint_manifest),
    "checkpoint_manifest_sha256": hashlib.sha256(
        checkpoint_manifest.read_bytes()
    ).hexdigest(),
    "cover_letter": str(cover_letter),
    "cover_letter_sha256": hashlib.sha256(cover_letter.read_bytes()).hexdigest(),
    "submission_package": str(submission_package),
    "submission_package_sha256": hashlib.sha256(
        submission_package.read_bytes()
    ).hexdigest(),
}
path = output / "paper_finalization.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
