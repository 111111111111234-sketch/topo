#!/usr/bin/env bash
# Walk one loop on the same floor, build topo map, and render visualization.
# Runs goat_agent_final and goat_agent_etpnav back-to-back on the same scene.
#
# Usage (VLM, default):
#   conda activate goat && bash scripts/run_loop_topo_viz_final_etpnav.sh
#
# Usage (CLIP + GroundingDINO):
#   PERCEPTION_BACKEND=clip_groundingdino bash scripts/run_loop_topo_viz_final_etpnav.sh
#
# Background:
#   nohup bash scripts/run_loop_topo_viz_final_etpnav.sh > data/logs/goat_topo/loop_viz_compare_vlm/nohup.out 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PERCEPTION_BACKEND="${PERCEPTION_BACKEND:-vlm}"
OUT="${OUT:-data/logs/goat_topo/loop_viz_compare_${PERCEPTION_BACKEND}}"
SPLIT="${SPLIT:-val_seen}"
SCENE="${SCENE:-4ok3usBNeis}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
VLM_API_BASE="${VLM_API_BASE:-http://localhost:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLM_TIMEOUT="${VLM_TIMEOUT:-60}"
HEAVY_ENABLED="${HEAVY_ENABLED:-1}"
LOOP_MAX_GEODESIC_LENGTH="${LOOP_MAX_GEODESIC_LENGTH:-40.0}"
VIZ_STRIDE="${VIZ_STRIDE:-2}"
VIZ_FPS="${VIZ_FPS:-6}"

eval "$(conda shell.bash hook)"
conda activate goat
export PYTHONPATH="$ROOT"

mkdir -p "$OUT"

run_one() {
  local agent="$1"
  local tag="$2"
  local run_dir="$OUT/${SCENE}_ep${EPISODE_INDEX}_${tag}"
  local trace="$run_dir/topo_trace_loop.json"
  local viz_dir="$run_dir/viz"
  local log="$run_dir/run.log"

  mkdir -p "$run_dir" "$viz_dir"
  echo "========================================" | tee "$log"
  echo "[$tag] loop + map build  $(date '+%F %T')" | tee -a "$log"
  echo "  agent=$agent  backend=$PERCEPTION_BACKEND  scene=$SCENE ep=$EPISODE_INDEX" | tee -a "$log"

  local heavy_args=()
  if [[ "$HEAVY_ENABLED" == "1" ]]; then
    heavy_args+=(--heavy-enabled)
  fi

  python scripts/run_goat_loop_topo_trace.py \
    --split "$SPLIT" \
    --scene "$SCENE" \
    --episode-index "$EPISODE_INDEX" \
    --agent "$agent" \
    --perception-backend "$PERCEPTION_BACKEND" \
    --vlm-api-base "$VLM_API_BASE" \
    --vlm-model "$VLM_MODEL" \
    --vlm-timeout "$VLM_TIMEOUT" \
    --clip-device "$CLIP_DEVICE" \
    "${heavy_args[@]}" \
    --loop-anchors 3 \
    --loop-max-geodesic-length "$LOOP_MAX_GEODESIC_LENGTH" \
    --loop-return-radius 0.75 \
    --output "$trace" \
    --frame-dir "$run_dir/rgb_frames" \
    2>&1 | tee -a "$log"

  python scripts/visualize_goat_topo_trace.py \
    --trace "$trace" \
    --out-dir "$viz_dir" \
    --stride "$VIZ_STRIDE" \
    --fps "$VIZ_FPS" \
    2>&1 | tee -a "$log"

  echo "[$tag] done" | tee -a "$log"
  echo "  trace: $trace" | tee -a "$log"
  echo "  video: $viz_dir/goat_semantic_audit_view.mp4" | tee -a "$log"
}

run_one final final
run_one etpnav etpnav

echo ""
echo "All done. backend=$PERCEPTION_BACKEND"
echo "  Final trace : $OUT/${SCENE}_ep${EPISODE_INDEX}_final/topo_trace_loop.json"
echo "  Final video : $OUT/${SCENE}_ep${EPISODE_INDEX}_final/viz/goat_semantic_audit_view.mp4"
echo "  ETPNav trace: $OUT/${SCENE}_ep${EPISODE_INDEX}_etpnav/topo_trace_loop.json"
echo "  ETPNav video: $OUT/${SCENE}_ep${EPISODE_INDEX}_etpnav/viz/goat_semantic_audit_view.mp4"
