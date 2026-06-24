#!/bin/bash
# GOAT + SOON val GoalGraphs (LLM + CLIP). Skips R2R and train splits.
#
# Terminal 1 (optional, if vLLM not already running):
#   bash ETPNav/scripts/start_vllm.sh 8000
#   # uses conda env vllm at /opt/conda/envs/vllm/bin/python
# Terminal 2:
#   conda activate goat && bash scripts/run_goal_graphs_goat_soon.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LLM_URL="${LLM_URL:-http://localhost:8000/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen2.5-7B-Instruct}"

eval "$(conda shell.bash hook)"
conda activate goat

echo "=== Step 1/2: LLM (GOAT + SOON only) ==="
python scripts/precompute_goal_graphs.py \
  --tasks goat soon \
  --no-train \
  --llm-url "$LLM_URL" \
  --llm-model "$LLM_MODEL"

echo ""
echo "=== Step 2/2: CLIP embed (text + landmarks + image goals) ==="
python scripts/precompute_goal_graphs.py \
  --embed-only \
  --no-train \
  --scene-root "${SCENE_ROOT:-data/scene_datasets/hm3d}"

echo ""
echo "Done. See data/goal_graphs/goat/ and data/goal_graphs/soon/"
