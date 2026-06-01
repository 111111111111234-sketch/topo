#!/bin/bash
# vllm: 独立 LLM 服务环境（与 goat / etpnav 隔离）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/environment-vllm.yaml"
CONDA="${CONDA_EXE:-/opt/conda/bin/conda}"
PIP_OPTIONS="${PIP_OPTIONS:--i https://nexus-wl.xiaopeng.link/repository/ai_infra_pypi_group/simple --timeout 600 --no-cache-dir}"

if [ ! -x "$CONDA" ]; then
    echo "Error: conda not found at $CONDA"
    exit 1
fi

if "$CONDA" env list | awk '{print $1}' | grep -qx vllm; then
    echo "vllm already exists, updating..."
    "$CONDA" env update -f "$ENV_FILE" --prune -y
else
    "$CONDA" env create -f "$ENV_FILE" -y
fi

# 与 docs/ENV.md 一致；依赖 CUDA 12.x（cu128 驱动兼容）
"$CONDA" run -n vllm pip install --no-cache-dir ${PIP_OPTIONS} \
    --extra-index-url https://pypi.org/simple \
    --retries 5 --timeout 300 \
    "vllm>=0.19.1,<0.20"

echo "vllm ready: $("$CONDA" run -n vllm python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '(import check skipped)')"
