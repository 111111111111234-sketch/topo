#!/bin/bash
# conftopo-dev: Core 开发 / 单测
# torch: PyPI 稳定版 2.8.0+cu128（非 NGC 镜像内 2.8.0a0）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/environment-dev.yaml"
CONDA="${CONDA_EXE:-/opt/conda/bin/conda}"
PIP_OPTIONS="${PIP_OPTIONS:--i https://nexus-wl.xiaopeng.link/repository/ai_infra_pypi_group/simple --timeout 600 --no-cache-dir}"

if [ ! -x "$CONDA" ]; then
    echo "Error: conda not found at $CONDA"
    exit 1
fi

if "$CONDA" env list | awk '{print $1}' | grep -qx conftopo-dev; then
    echo "conftopo-dev already exists, updating..."
    "$CONDA" env update -f "$ENV_FILE" --prune -y
else
    "$CONDA" env create -f "$ENV_FILE" -y
fi

"$CONDA" run -n conftopo-dev pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "conftopo-dev ready: $("$CONDA" run -n conftopo-dev python --version)"
