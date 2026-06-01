#!/bin/bash
# Verify conftopo-dev / vllm / goat / etpnav after image build
set -euo pipefail

DEV_PY="${CONFTopo_DEV_PYTHON:-/opt/conda/envs/conftopo-dev/bin/python}"
VLLM_PY="${CONFTopo_VLLM_PYTHON:-/opt/conda/envs/vllm/bin/python}"
CONDA="/opt/conda/bin/conda"
FAIL=0

pass() { echo "  ✓ $*"; }
fail() { echo "  ✗ $*"; FAIL=1; }
skip() { echo "  ○ $* (not yet built)"; }

echo "========== ConfTopo Environment Verification =========="
echo ""

echo "[1] conftopo-dev (Core / 单测) — $DEV_PY"
if "$DEV_PY" -c "
import torch
from conftopo import ConfTopoConfig
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import openai
print('openai', openai.__version__)
" 2>/dev/null; then
    pass "conftopo-dev"
else
    fail "conftopo-dev"
fi
echo ""

echo "[2] vllm (LLM 服务) — $VLLM_PY"
if ! "$CONDA" env list | awk '{print $1}' | grep -qx vllm; then
    skip "vllm (run Dockerfile.part2)"
elif "$VLLM_PY" -c "
import vllm
print('vllm', vllm.__version__)
" 2>/dev/null; then
    pass "vllm environment"
else
    fail "vllm environment"
fi
echo ""

echo "[3] goat — ConfTopo-GOAT / habitat-sim 0.2.4"
if ! "$CONDA" env list | awk '{print $1}' | grep -qx goat; then
    skip "goat (run Dockerfile.part3)"
elif env -u LD_LIBRARY_PATH "$CONDA" run -n goat python -c "
import torch, habitat_sim
from conftopo import ConfTopoConfig
print('torch', torch.__version__, 'habitat_sim', habitat_sim.__version__)
try:
    import habitat
    print('habitat', habitat.__version__)
except ImportError:
    print('habitat-lab not installed yet')
" 2>/dev/null; then
    pass "goat environment"
else
    fail "goat environment"
fi
echo ""

echo "[4] vlnce/etpnav — ConfTopo-ETPNav / habitat-sim 0.1.7 + torch 2.8+cu128"
ETPNAV_ENV=""
if "$CONDA" env list | awk '{print $1}' | grep -qx vlnce; then
    ETPNAV_ENV="vlnce"
elif "$CONDA" env list | awk '{print $1}' | grep -qx etpnav; then
    ETPNAV_ENV="etpnav"
fi

if [ -z "$ETPNAV_ENV" ]; then
    skip "vlnce / etpnav (run unified Dockerfile or Dockerfile.part4)"
elif env -u LD_LIBRARY_PATH "$CONDA" run -n "$ETPNAV_ENV" python -c "
import torch, habitat_sim, gym, transformers
from conftopo import ConfTopoConfig
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('habitat_sim', habitat_sim.__version__, 'gym', gym.__version__)
import habitat
print('habitat', habitat.__version__)
" 2>/dev/null; then
    pass "$ETPNAV_ENV environment"
else
    fail "$ETPNAV_ENV environment"
fi
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "========== All required environments OK =========="
else
    echo "========== Some checks failed =========="
    exit 1
fi
