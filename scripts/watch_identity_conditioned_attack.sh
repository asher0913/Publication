#!/usr/bin/env bash
set -u

ROOT="${PUBLICATION_ROOT:-/home/unnc/zhang/Publication}"
PYTHON="${PUBLICATION_PYTHON:-/home/unnc/miniconda3/envs/publication-cem/bin/python}"
DATA_ROOT="${PUBLICATION_DATA_ROOT:-/home/unnc/zhang/attention-face/data/facescrub}"
TARGETS_ROOT="${IDENTITY_ATTACK_TARGETS_ROOT:-$ROOT/results/dual_path_publication/targets}"
LOG_ROOT="$ROOT/run_logs/identity_conditioned_attack"
MASTER_LOG="$LOG_ROOT/watchdog.log"
LOCK="$ROOT/run_state/identity_conditioned_attack.lock"
SUMMARY="$ROOT/results/identity_conditioned_attack/identity_conditioned_attack_summary.json"

mkdir -p "$LOG_ROOT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "IDENTITY_ATTACK_WATCHDOG_ALREADY_RUNNING"
    exit 0
fi

summary_complete() {
    "$PYTHON" - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "PASS" else 1)
PY
}

cd "$ROOT"
while true; do
    if summary_complete; then
        echo "$(date --iso-8601=seconds) IDENTITY_ATTACK_EVIDENCE_READY" \
            | tee -a "$MASTER_LOG"
        exit 0
    fi
    echo "$(date --iso-8601=seconds) IDENTITY_ATTACK_PIPELINE_START" \
        | tee -a "$MASTER_LOG"
    "$PYTHON" scripts/run_identity_conditioned_attack_protocol.py \
        --data-root "$DATA_ROOT" \
        --targets-root "$TARGETS_ROOT" \
        --attempts 3 \
        --retry-seconds 60 >>"$MASTER_LOG" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]] && summary_complete; then
        continue
    fi
    echo "$(date --iso-8601=seconds) IDENTITY_ATTACK_RESTART rc=$rc" \
        | tee -a "$MASTER_LOG"
    sleep 120
done
