#!/bin/bash
# Generate val GoalGraphs with LLM + CLIP (skip train / large splits).
#
# Two conda environments:
#   vllm_server  — inference service only (GPU)
#   conftopo     — this script (HTTP client + open_clip)
#
# Terminal 1 — start vLLM (keep running):
#   conda activate vllm_server
#   bash ETPNav/scripts/start_vllm.sh 8000
#
# Terminal 2 — precompute:
#   bash scripts/run_goal_graphs_val.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LLM_URL="${LLM_URL:-http://localhost:8000/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen2.5-7B-Instruct}"

eval "$(conda shell.bash hook)"
conda activate conftopo

echo "=== Step 1/2: LLM structuring (HTTP -> vllm_server) ==="
python scripts/precompute_goal_graphs.py \
  --task all \
  --no-train \
  --llm-url "$LLM_URL" \
  --llm-model "$LLM_MODEL"

echo ""
echo "=== Step 2/2: CLIP embeddings (conftopo / open_clip) ==="
python scripts/precompute_goal_graphs.py \
  --embed-only \
  --no-train

echo ""
echo "Done. Outputs under data/goal_graphs/{r2r,goat,soon}/"
