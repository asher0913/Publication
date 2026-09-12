#!/usr/bin/env bash
set -u

ROOT="${PUBLICATION_ROOT:-/home/unnc/zhang/Publication}"
PYTHON="${PUBLICATION_PYTHON:-/home/unnc/miniconda3/envs/publication-cem/bin/python}"
MASTER_SESSION="${PUBLICATION_MASTER_SESSION:-pr-dual-formal-20260804}"
STATE="$ROOT/run_state/dual_path_publication.json"
LOG="$ROOT/run_logs/dual_path_publication_master_20260804.log"
DATA_ROOT="${PUBLICATION_DATA_ROOT:-/home/unnc/zhang/attention-face/data/facescrub}"
LEGACY_CHECKPOINT="${PUBLICATION_LEGACY_CHECKPOINT:-/home/unnc/zhang/attention-face/cem-fixed/saves/facescrub/SCA_new_cemfixed_acc80c16v2_vt0.15/l10_n0.025_ep300_vt0.15_ls1.0_sl8_it3_bk64_sd64_wu3_at0.15_bnnoRELU_C16S1}"

json_status() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    print("MISSING")
else:
    print(payload.get("status", "UNKNOWN"))
PY
}

scientific_gate_failed() {
    "$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "results" / "dual_path_publication"
paths = list(root.glob("targets/seed*/utility_l031_s010.json"))
paths.append(root / "formal_summary.json")
for path in paths:
    if not path.is_file():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue
    if payload.get("status") == "FAIL":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

cd "$ROOT"
while tmux has-session -t "$MASTER_SESSION" 2>/dev/null; do
    sleep 30
done

while true; do
    if [[ "$(json_status "$STATE")" == "PASS" ]]; then
        echo "FORMAL_PIPELINE_COMPLETE_FINALISING_PAPER"
        if ! git pull --ff-only origin codex/pareto-training-control; then
            sleep 120
            continue
        fi
        for attempt in 1 2 3; do
            if bash scripts/finalize_pattern_recognition_paper.sh; then
                echo "FORMAL_PIPELINE_AND_PAPER_COMPLETE"
                exit 0
            fi
            echo "PAPER_FINALISATION_RETRY attempt=$attempt" >>"$LOG"
            sleep 60
        done
        echo "PAPER_FINALISATION_FAILED" >>"$LOG"
        exit 1
    fi
    if scientific_gate_failed; then
        echo "SCIENTIFIC_GATE_FAIL_PRESERVED"
        exit 2
    fi

    if ! git pull --ff-only origin codex/pareto-training-control; then
        echo "WATCHDOG_GIT_PULL_RETRY_AFTER_120_SECONDS" >>"$LOG"
        sleep 120
        continue
    fi
    "$PYTHON" scripts/run_dual_path_publication_pipeline.py \
        --data-root "$DATA_ROOT" \
        --legacy-checkpoint-dir "$LEGACY_CHECKPOINT" \
        --stage all \
        --attempts 3 \
        --retry-seconds 60 >>"$LOG" 2>&1
    rc=$?
    if [[ $rc -eq 0 || $rc -eq 2 ]]; then
        exit "$rc"
    fi
    echo "MASTER_RESTART_AFTER_120_SECONDS rc=$rc" >>"$LOG"
    sleep 120
done
