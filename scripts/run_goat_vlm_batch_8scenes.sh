#!/bin/bash
# GOAT VLM batch eval: 8 val_seen scenes × 10 goals × 500 steps/goal.
#
# Prereqs:
#   - conda env `goat`
#   - vLLM serving Qwen3-VL-8B on VLM_API_BASE (default :8000)
#   - GoalGraphs + HM3D scenes under data/
#
# Foreground:
#   conda activate goat && bash scripts/run_goat_vlm_batch_8scenes.sh
#
# Background:
#   nohup bash scripts/run_goat_vlm_batch_8scenes.sh > data/logs/goat_topo/vlm_batch_8xfullx500/nohup.out 2>&1 &
#   tail -f data/logs/goat_topo/vlm_batch_8xfullx500/run.log

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-data/logs/goat_topo/vlm_batch_8xfullx500}"
SPLIT="${SPLIT:-val_seen}"
MAX_GOALS="${MAX_GOALS:-10}"
STEPS_PER_GOAL="${STEPS_PER_GOAL:-500}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-1.0}"

VLM_API_BASE="${VLM_API_BASE:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLM_TIMEOUT="${VLM_TIMEOUT:-60}"

HEAVY_INTERVAL="${HEAVY_INTERVAL:-}"
HEAVY_REGROUND_COOLDOWN="${HEAVY_REGROUND_COOLDOWN:-}"
HEAVY_NEAR_GOAL_COOLDOWN="${HEAVY_NEAR_GOAL_COOLDOWN:-}"

CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
CLIP_IMAGE_MODEL="${CLIP_IMAGE_MODEL:-RN50}"
CLIP_DEVICE="${CLIP_DEVICE:-cuda}"

DATASET_DIR="${DATASET_DIR:-data/datasets/goat_bench/hm3d/v1}"
SCENE_ROOT="${SCENE_ROOT:-data/scene_datasets/hm3d}"
GOAL_GRAPH_DIR="${GOAL_GRAPH_DIR:-data/goal_graphs/goat}"

# scene:episode_index — each episode has 10 goals on val_seen
RUNS=(
  "4ok3usBNeis:0"
  "5cdEh9F2hJL:7"
  "6s7QHgap2fW:7"
  "7MXmsvcQjpJ:2"
  "BAbdmeyTvMZ:3"
  "CrMo8WxCyVb:1"
  "DYehNKdT76V:0"
  "Dd4bFSTQ8gi:6"
)

mkdir -p "$OUT"
LOG="$OUT/run.log"
LOCK="$OUT/.batch.lock"
PIDFILE="$OUT/.batch.pid"

# Prevent duplicate batch runs (e.g. nohup twice while first scene still loading).
exec 9>"$LOCK"
if ! flock -n 9; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || echo unknown)"
  echo "ERROR: batch already running (pid=$old_pid). Stop it first:"
  echo "  kill $old_pid"
  echo "  # or: pkill -f run_goat_vlm_batch_8scenes.sh"
  exit 1
fi
echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

: >"$LOG"   # fresh log for this batch session

eval "$(conda shell.bash hook)"
conda activate goat
export PYTHONPATH="$ROOT"

echo "========================================" | tee -a "$LOG"
echo "GOAT VLM batch start  $(date '+%F %T')" | tee -a "$LOG"
echo "  OUT=$OUT" | tee -a "$LOG"
echo "  SPLIT=$SPLIT  goals=$MAX_GOALS  steps/goal=$STEPS_PER_GOAL" | tee -a "$LOG"
echo "  VLM=$VLM_MODEL @ $VLM_API_BASE" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

if ! curl -sf "${VLM_API_BASE}/models" | grep -q "Qwen3-VL"; then
  echo "WARN: VLM not reachable or model name mismatch at $VLM_API_BASE" | tee -a "$LOG"
  echo "      Start vLLM first, then re-run." | tee -a "$LOG"
fi

EXTRA_ARGS=()
if [[ -n "$HEAVY_INTERVAL" ]]; then
  EXTRA_ARGS+=(--heavy-interval "$HEAVY_INTERVAL")
fi
if [[ -n "$HEAVY_REGROUND_COOLDOWN" ]]; then
  EXTRA_ARGS+=(--heavy-reground-cooldown "$HEAVY_REGROUND_COOLDOWN")
fi
if [[ -n "$HEAVY_NEAR_GOAL_COOLDOWN" ]]; then
  EXTRA_ARGS+=(--heavy-near-goal-cooldown "$HEAVY_NEAR_GOAL_COOLDOWN")
fi

FAILED=()
i=0
total="${#RUNS[@]}"

for item in "${RUNS[@]}"; do
  scene="${item%%:*}"
  ep="${item##*:}"
  i=$((i + 1))
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "[$i/$total] scene=$scene episode=$ep  $(date '+%F %T')" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"

  if python scripts/run_goat_multigoal_acceptance.py \
    --split "$SPLIT" \
    --scene "$scene" \
    --episode-index "$ep" \
    --max-goals "$MAX_GOALS" \
    --steps-per-goal "$STEPS_PER_GOAL" \
    --success-distance "$SUCCESS_DISTANCE" \
    --perception-backend vlm \
    --vlm-api-base "$VLM_API_BASE" \
    --vlm-model "$VLM_MODEL" \
    --vlm-timeout "$VLM_TIMEOUT" \
    --clip-model "$CLIP_MODEL" \
    --clip-image-model "$CLIP_IMAGE_MODEL" \
    --clip-device "$CLIP_DEVICE" \
    --dataset-dir "$DATASET_DIR" \
    --scene-root "$SCENE_ROOT" \
    --goal-graph-dir "$GOAL_GRAPH_DIR" \
    --output "$OUT/${scene}_ep${ep}_trace.json" \
    --report "$OUT/${scene}_ep${ep}_report.json" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"; then
    echo "[$i/$total] OK scene=$scene  $(date '+%F %T')" | tee -a "$LOG"
  else
    echo "[$i/$total] FAILED scene=$scene  $(date '+%F %T')" | tee -a "$LOG"
    FAILED+=("$scene")
  fi
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Batch finished  $(date '+%F %T')" | tee -a "$LOG"
if ((${#FAILED[@]} > 0)); then
  echo "Failed scenes (${#FAILED[@]}): ${FAILED[*]}" | tee -a "$LOG"
else
  echo "All $total scenes completed." | tee -a "$LOG"
fi
echo "========================================" | tee -a "$LOG"

python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json
import glob
import sys
from pathlib import Path

out = Path(sys.argv[1])
reports = sorted(glob.glob(str(out / "*_report.json")))
if not reports:
    print("No report files found for summary.")
    raise SystemExit(0)

succ = total = 0
print(f"{'scene':<16} {'SR':<8} {'pipe':<6} {'stopped':<10} avg_min_dist")
print("-" * 55)
for p in reports:
    scene = Path(p).name.split("_ep")[0]
    r = json.load(open(p))
    goals = r.get("task_summaries", [])
    s = sum(1 for g in goals if g.get("goal_success"))
    st = sum(1 for g in goals if g.get("goal_stopped"))
    ds = [g["goal_min_distance"] for g in goals if g.get("goal_min_distance") is not None]
    avg = sum(ds) / len(ds) if ds else float("nan")
    succ += s
    total += len(goals)
    avg_s = f"{avg:.1f}m" if ds else "n/a"
    print(f"{scene:<16} {s}/{len(goals):<6} {str(r.get('overall_passed')):<6} {st}/{len(goals):<8} {avg_s}")
print("-" * 55)
print(f"Total SR: {succ}/{total} ({100 * succ / max(total, 1):.1f}%)")
PY

if ((${#FAILED[@]} > 0)); then
  exit 1
fi
