#!/usr/bin/env bash
set -uo pipefail

ROOT="${PUBLICATION_ROOT:-/home/unnc/zhang/Publication}"
PYTHON="${PUBLICATION_PYTHON:-/home/unnc/miniconda3/envs/publication-cem/bin/python}"
DATA_ROOT="${PUBLICATION_DATA_ROOT:-/home/unnc/zhang/attention-face/data/facescrub}"
LEGACY_CHECKPOINT="${PUBLICATION_LEGACY_CHECKPOINT:-/home/unnc/zhang/attention-face/cem-fixed/saves/facescrub/SCA_new_cemfixed_acc80c16v2_vt0.15/l10_n0.025_ep300_vt0.15_ls1.0_sl8_it3_bk64_sd64_wu3_at0.15_bnnoRELU_C16S1}"
STATE="$ROOT/run_state/dual_path_supplementary.json"
LOG_ROOT="$ROOT/run_logs/dual_path_supplementary"
POLL_SECONDS="${PUBLICATION_WATCHDOG_POLL_SECONDS:-60}"
MAX_RESTARTS="${PUBLICATION_WATCHDOG_MAX_RESTARTS:-10}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1

campaign_passed() {
    "$PYTHON" - "$STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
state = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if state.get("status") == "PASS" else 1)
PY
}

pipeline_running() {
    pgrep -u "$(id -u)" -f \
        "[r]un_dual_path_supplementary_pipeline.py.*--stage all" >/dev/null
}

for restart in $(seq 0 "$MAX_RESTARTS"); do
    while pipeline_running; do
        sleep "$POLL_SECONDS"
    done
    if campaign_passed; then
        printf '%s campaign complete\n' "$(date --iso-8601=seconds)"
        exit 0
    fi
    if (( restart == MAX_RESTARTS )); then
        printf '%s restart limit reached\n' "$(date --iso-8601=seconds)" >&2
        exit 2
    fi
    log="$LOG_ROOT/watchdog_recovery_$(date +%Y%m%d_%H%M%S).log"
    printf '%s restarting campaign (%d/%d)\n' \
        "$(date --iso-8601=seconds)" "$((restart + 1))" "$MAX_RESTARTS" | \
        tee -a "$log"
    "$PYTHON" scripts/run_dual_path_supplementary_pipeline.py \
        --data-root "$DATA_ROOT" \
        --legacy-checkpoint-dir "$LEGACY_CHECKPOINT" \
        --stage all \
        --attempts 3 \
        --retry-seconds 60 >>"$log" 2>&1
done
