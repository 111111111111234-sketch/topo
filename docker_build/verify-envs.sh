#!/bin/bash
# Verify core / vllm / goat / etpnav environments after image build
set -euo pipefail

MAIN_PY="${CONFTopo_MAIN_PYTHON:-/opt/conda/envs/conftopo-main/bin/python}"
DEV_PY="${CONFTopo_DEV_PYTHON:-/opt/conda/envs/conftopo-dev/bin/python}"
VLLM_PY="${CONFTopo_VLLM_PYTHON:-/opt/conda/envs/vllm/bin/python}"
CONDA="/opt/conda/bin/conda"
FAIL=0
VERIFY_ONLY="${VERIFY_ONLY:-all}"

pass() { echo "  ✓ $*"; }
fail() { echo "  ✗ $*"; FAIL=1; }
skip() { echo "  ○ $* (not yet built)"; }

run_conda_python_check() {
    local env_name="$1"
    local label="$2"
    local code="$3"
    local extra_ld_path="${4:-}"

    echo "  - ${label}"
    if [ -n "$extra_ld_path" ]; then
        LD_LIBRARY_PATH="${extra_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "$CONDA" run -n "$env_name" python -c "$code"
    else
        env -u LD_LIBRARY_PATH "$CONDA" run -n "$env_name" python -c "$code"
    fi
}

get_habitat_sim_ext_dir() {
    local env_name="$1"

    "$CONDA" run -n "$env_name" python -c "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()['purelib']) / 'habitat_sim' / '_ext')" | tail -n 1
}

should_check() {
    if [ "$VERIFY_ONLY" = "all" ]; then
        return 0
    fi
    case ",${VERIFY_ONLY}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

echo "========== ConfTopo Environment Verification =========="
echo ""

if should_check conftopo-main || should_check conftopo-dev; then
CORE_ENV=""
CORE_PY=""
if "$CONDA" env list | awk '{print $1}' | grep -qx conftopo-main; then
    CORE_ENV="conftopo-main"
    CORE_PY="$MAIN_PY"
elif "$CONDA" env list | awk '{print $1}' | grep -qx conftopo-dev; then
    CORE_ENV="conftopo-dev"
    CORE_PY="$DEV_PY"
fi

echo "[1] core (conftopo-main / conftopo-dev)"
if [ -z "$CORE_ENV" ]; then
    skip "conftopo-main / conftopo-dev"
elif "$CORE_PY" -c "
import torch
from conftopo import ConfTopoConfig
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import openai
print('openai', openai.__version__)
" 2>/dev/null; then
    pass "$CORE_ENV"
else
    fail "$CORE_ENV"
fi
echo ""
fi

if should_check vllm; then
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
fi

if should_check goat; then
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
fi

if should_check vlnce; then
echo "[4] vlnce/etpnav — ConfTopo-ETPNav / habitat-sim 0.1.7 + torch 2.8+cu128"
ETPNAV_ENV=""
if "$CONDA" env list | awk '{print $1}' | grep -qx vlnce; then
    ETPNAV_ENV="vlnce"
elif "$CONDA" env list | awk '{print $1}' | grep -qx etpnav; then
    ETPNAV_ENV="etpnav"
fi

if [ -z "$ETPNAV_ENV" ]; then
    skip "vlnce / etpnav (run unified Dockerfile or Dockerfile.part4)"
else
    ETPNAV_EXT_DIR="$(get_habitat_sim_ext_dir "$ETPNAV_ENV")"
    ETPNAV_EXTRA_LD=""
    if [ -d "$ETPNAV_EXT_DIR" ]; then
        ETPNAV_EXTRA_LD="${ETPNAV_EXT_DIR}:/opt/conda/envs/${ETPNAV_ENV}/lib"
    fi

    if run_conda_python_check "$ETPNAV_ENV" "torch" "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'torch_cuda', torch.version.cuda)
" && \
        run_conda_python_check "$ETPNAV_ENV" "habitat_sim" "
import habitat_sim
print('habitat_sim', habitat_sim.__version__)
" "$ETPNAV_EXTRA_LD" && \
        run_conda_python_check "$ETPNAV_ENV" "habitat" "
import habitat
print('habitat', habitat.__version__)
" "$ETPNAV_EXTRA_LD" && \
        run_conda_python_check "$ETPNAV_ENV" "conftopo + gym + transformers" "
import gym, transformers
from conftopo import ConfTopoConfig
print('gym', gym.__version__, 'transformers', transformers.__version__)
print('conftopo', ConfTopoConfig.__name__)
" "$ETPNAV_EXTRA_LD"; then
        pass "$ETPNAV_ENV environment"
    else
        fail "$ETPNAV_ENV environment"
    fi
fi
echo ""
fi

if [ "$FAIL" -eq 0 ]; then
    echo "========== All required environments OK =========="
else
    echo "========== Some checks failed =========="
    exit 1
fi
