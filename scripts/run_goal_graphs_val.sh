#!/bin/bash
# Generate val GoalGraphs with LLM + CLIP (skip train / large splits).
#
# conda env: goat (LLM client + open_clip + habitat)
#
# Terminal 1 — start vLLM (keep running, optional if already up):
#   conda activate goat
#   bash ETPNav/scripts/start_vllm.sh 8000
#
# Terminal 2 — precompute:
#   conda activate goat && bash scripts/run_goal_graphs_val.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LLM_URL="${LLM_URL:-http://localhost:8000/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen2.5-7B-Instruct}"

eval "$(conda shell.bash hook)"
conda activate goat

echo "=== Step 1/2: LLM structuring (HTTP -> vllm_server) ==="
python scripts/precompute_goal_graphs.py \
  --task all \
  --no-train \
  --llm-url "$LLM_URL" \
  --llm-model "$LLM_MODEL"

echo ""
echo "=== Step 2/2: CLIP embeddings (goat / open_clip) ==="
python scripts/precompute_goal_graphs.py \
  --embed-only \
  --no-train \
  --scene-root "${SCENE_ROOT:-data/scene_datasets/hm3d}"

echo ""
echo "Done. Outputs under data/goal_graphs/{r2r,goat,soon}/"
