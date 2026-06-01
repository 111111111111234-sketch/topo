#!/bin/bash
# Start vLLM inference server (conda env: vllm)
# Usage: start-vllm.sh [PORT] [GPU_UTIL]

set -euo pipefail

PORT="${1:-8000}"
GPU_UTIL="${2:-0.25}"
WORKSPACE="${WORKSPACE:-/workspace/tangyx7@xiaopeng.com}"
MODEL_PATH="${VLLM_MODEL_PATH:-${WORKSPACE}/models/Qwen3-4B}"
# 新版: vllm conda；旧版 part1: 系统 python（CONFTopo_MAIN_PYTHON）
PYTHON="${CONFTopo_VLLM_PYTHON:-${CONFTopo_MAIN_PYTHON:-/opt/conda/envs/vllm/bin/python}}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Download example:"
    echo "  modelscope download Qwen/Qwen3-4B --local_dir $MODEL_PATH"
    exit 1
fi

echo "Starting vLLM server..."
echo "  Python: $PYTHON (env: vllm)"
echo "  Model:  $MODEL_PATH"
echo "  Port:   $PORT"
echo "  GPU util: $GPU_UTIL"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen3-4B" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 4096 \
    --port "$PORT" \
    --dtype bfloat16
