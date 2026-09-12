#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-$ROOT/submission/data/facescrub_official}"
LEGACY_CHECKPOINT_DIR="${LEGACY_CHECKPOINT_DIR:-$ROOT/submission/results/legacy_slot_cem_checkpoint}"
TARGETS_ROOT="${TARGETS_ROOT:-$ROOT/results/dual_path_publication/targets}"
if [[ ! -d "$TARGETS_ROOT" && -d "$ROOT/submission/results/full_runs/dual_path_publication/targets" ]]; then
  TARGETS_ROOT="$ROOT/submission/results/full_runs/dual_path_publication/targets"
fi
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

while true; do
  python scripts/run_supervisor_revision_experiments.py \
    --data-root "$DATA_ROOT" \
    --legacy-checkpoint-dir "$LEGACY_CHECKPOINT_DIR" \
    --targets-root "$TARGETS_ROOT" \
    --attempts 3 \
    --retry-seconds 60
  status=$?
  if [[ $status -eq 0 ]]; then
    exit 0
  fi
  printf '[%s] campaign exited with status %s; retrying in %ss\n' \
    "$(date --iso-8601=seconds)" "$status" "$SLEEP_SECONDS" >&2
  sleep "$SLEEP_SECONDS"
done
