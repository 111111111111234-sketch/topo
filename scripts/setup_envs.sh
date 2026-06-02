#!/bin/bash
# ConfTopo Project: Environment Setup Script
# 基于 docs/ENV.md 的配置复现
#
# Usage: 
#   bash scripts/setup_envs.sh all        # 安装所有环境
#   bash scripts/setup_envs.sh vlnce      # 只安装 vlnce (官方 ETPNav baseline)
#   bash scripts/setup_envs.sh conftopo   # 只安装 conftopo (开发主环境)
#   bash scripts/setup_envs.sh goat       # 只安装 goat (GOAT-Bench)
#   bash scripts/setup_envs.sh vllm       # 只安装 vllm_server

set -e
WORKSPACE="/workspace/tangyx7@xiaopeng.com"
PIP_INDEX="https://pypi.org/simple/"
# PyTorch 国内镜像 (阿里云)
TORCH_MIRROR="https://mirrors.aliyun.com/pytorch-wheels"

# ============================================================
# vlnce: ETPNav 官方 baseline (Python 3.8)
#   - habitat-sim 0.1.7 (conda pre-built)
#   - torch 1.9.1+cu111
#   - habitat-lab 0.1.7
# ============================================================
setup_vlnce() {
    echo "============================================"
    echo "[vlnce] ETPNav 官方 baseline"
    echo "  Python 3.8 + habitat-sim 0.1.7 + torch 1.9.1+cu111"
    echo "============================================"
    
    conda remove -n vlnce --all -y 2>/dev/null || true
    conda env create -f "$WORKSPACE/ETPNav/environment.yaml" || {
        echo "environment.yaml 创建失败，手动创建..."
        conda create -n vlnce python=3.8 -y
        conda install -n vlnce habitat-sim=0.1.7 headless -c aihabitat -c conda-forge -y
    }
    
    # PyTorch 1.9.1 + CUDA 11.1
    conda run -n vlnce pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html -i "$PIP_INDEX" || \
        echo "⚠️ torch 1.9.1 下载太慢，可改用 conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch"
    
    # habitat-lab 0.1.7
    cd "$WORKSPACE/habitat-lab"
    conda run -n vlnce pip install -e . --no-deps
    
    # CLIP
    conda run -n vlnce pip install git+https://github.com/openai/CLIP.git -i "$PIP_INDEX" || true
    conda run -n vlnce pip install gym==0.21.0 -i "$PIP_INDEX"
    
    echo "✓ vlnce environment ready"
}

# ============================================================
# conftopo: 升级后的 ETPNav + ConfTopo 开发环境 (Python 3.9)
#   - habitat-sim 0.1.7 (源码编译)
#   - PyTorch 2.8.0+cu128
#   - habitat-lab 0.1.7
#   - open_clip, conftopo
# ============================================================
setup_conftopo() {
    echo "============================================"
    echo "[conftopo] ConfTopo 开发主环境"
    echo "  Python 3.9 + PyTorch 2.8+cu128 + habitat-sim 0.1.7 (源码)"
    echo "============================================"
    
    conda remove -n conftopo --all -y 2>/dev/null || true
    conda create -n conftopo python=3.9 cmake -y
    
    # Fix pip: conda may install pip 26+ which requires Python 3.10+
    conda run -n conftopo python -c "
import os, shutil, glob
for d in glob.glob('/opt/conda/envs/conftopo/lib/python3.9/site-packages/pip-2[4-9]*') + \
         glob.glob('/opt/conda/envs/conftopo/lib/python3.9/site-packages/pip-[3-9]*'):
    shutil.rmtree(d)
pip_dir = '/opt/conda/envs/conftopo/lib/python3.9/site-packages/pip'
if os.path.exists(pip_dir):
    shutil.rmtree(pip_dir)
" 2>/dev/null || true
    conda run -n conftopo python -m ensurepip --upgrade 2>/dev/null
    conda run -n conftopo pip install "pip>=23,<24" -i "$PIP_INDEX" 2>/dev/null || true
    
    # PyTorch 2.8 + CUDA 12.8 (Blackwell GPU 必须 2.8+)
    # 国内镜像暂无 2.8，只能走官方源（约 2.5GB）
    conda run -n conftopo pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
    
    # Core dependencies
    conda run -n conftopo pip install \
        numpy==1.26.4 \
        scipy \
        scikit-learn \
        matplotlib==3.9.4 \
        opencv-python==4.11.0.86 \
        networkx==3.2.1 \
        Pillow \
        tqdm \
        pyyaml \
        -i "$PIP_INDEX"
    
    # Habitat / ETPNav deps
    conda run -n conftopo pip install \
        gym==0.23.1 \
        hydra-core \
        omegaconf \
        yacs \
        imageio imageio-ffmpeg \
        quaternion \
        jsonlines \
        h5py \
        lmdb \
        msgpack msgpack-numpy \
        timm \
        transformers==4.12.5 \
        tensorboardx \
        ftfy regex \
        easydict \
        -i "$PIP_INDEX"
    
    # ConfTopo 额外依赖
    conda run -n conftopo pip install \
        open_clip_torch==3.3.0 \
        openai \
        safetensors \
        huggingface_hub \
        gdown \
        requests \
        -i "$PIP_INDEX"
    
    # habitat-sim 0.1.7 源码编译
    echo ""
    echo "⚠️  habitat-sim 0.1.7 需要源码编译 (Python 3.9 无预编译包)"
    echo "    参考: https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md"
    echo "    编译命令 (需在 conftopo 环境中):"
    echo "      cd /path/to/habitat-sim-v0.1.7"
    echo "      pip install -r requirements.txt"
    echo "      python setup.py install --headless --with-cuda"
    echo ""
    
    # habitat-lab 0.1.7
    cd "$WORKSPACE/habitat-lab"
    conda run -n conftopo pip install -e . --no-deps
    
    # conftopo 包
    cd "$WORKSPACE"
    conda run -n conftopo pip install -e . 2>/dev/null || true
    
    # CLIP
    conda run -n conftopo pip install git+https://github.com/openai/CLIP.git || true
    
    echo "✓ conftopo environment ready (除 habitat-sim 源码编译外)"
}

# ============================================================
# goat: GOAT-Bench 评估环境 (Python 3.8)
#   - habitat-sim 0.2.3
#   - habitat-lab 0.2.3
# ============================================================
setup_goat() {
    echo "============================================"
    echo "[goat] GOAT-Bench 评估"
    echo "  Python 3.8 + habitat-sim 0.2.3 + habitat-lab 0.2.3"
    echo "============================================"
    
    conda remove -n goat --all -y 2>/dev/null || true
    conda create -n goat python=3.8 cmake=3.14.0 -y
    
    # habitat-sim 0.2.3
    conda install -n goat habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y
    
    # PyTorch (via conda)
    conda install -n goat pytorch torchvision cudatoolkit=11.3 -c pytorch -c nvidia -y
    
    # OpenCV + scikit-learn
    conda install -n goat opencv scikit-learn pyyaml -c conda-forge -y
    
    # habitat-lab 0.2.3
    conda run -n goat pip install pytest-runner -i "$PIP_INDEX"
    cd "$WORKSPACE/habitat-lab-v0.2.3"
    conda run -n goat pip install -e habitat-lab --no-deps
    
    # Core pip deps
    conda run -n goat pip install \
        hydra-core omegaconf \
        yacs ifcfg \
        "gym>=0.22,<0.24" \
        imageio imageio-ffmpeg \
        tqdm \
        jsonlines \
        networkx \
        ftfy regex \
        einops \
        -i "$PIP_INDEX"
    
    # conftopo 包
    cd "$WORKSPACE"
    conda run -n goat pip install -e . 2>/dev/null || true
    
    echo "✓ goat environment ready"
}

# ============================================================
# vllm_server: LLM 推理 (Python 3.10)
#   - vLLM 0.19.1
#   - 模型: Qwen2.5-7B-Instruct
# ============================================================
setup_vllm() {
    echo "============================================"
    echo "[vllm_server] LLM 推理服务"
    echo "  Python 3.10 + vLLM 0.19.1"
    echo "============================================"
    
    conda remove -n vllm_server --all -y 2>/dev/null || true
    conda create -n vllm_server python=3.10 -y
    
    # vLLM 固定版本 (0.21+ 需要 CUDA 13)
    conda run -n vllm_server pip install vllm==0.19.1 -i "$PIP_INDEX"
    
    echo ""
    echo "  模型路径: /root/workspace/models/Qwen2.5-7B-Instruct"
    echo "  启动命令: conda activate vllm_server && bash ETPNav/scripts/start_vllm.sh"
    echo ""
    echo "✓ vllm_server environment ready"
}

# ============================================================
# Main
# ============================================================
TARGET="${1:-all}"

case "$TARGET" in
    vlnce)    setup_vlnce ;;
    conftopo) setup_conftopo ;;
    goat)     setup_goat ;;
    vllm)     setup_vllm ;;
    all)
        setup_conftopo
        setup_goat
        setup_vllm
        echo ""
        echo "============================================"
        echo "环境创建完成！"
        echo "============================================"
        echo ""
        echo "  conda activate conftopo    # 开发主环境 (ETPNav + ConfTopo)"
        echo "  conda activate goat        # GOAT-Bench 评估"
        echo "  conda activate vllm_server # LLM 推理服务"
        echo ""
        echo "⚠️  注意事项:"
        echo "  - conftopo 环境需要手动编译 habitat-sim 0.1.7"
        echo "  - vlnce 环境未包含在 all 中 (官方 baseline 备选，一般用 conftopo 代替)"
        echo "  - vllm_server 固定 0.19.1，不要升级"
        ;;
    *)
        echo "Usage: bash scripts/setup_envs.sh {vlnce|conftopo|goat|vllm|all}"
        exit 1
        ;;
esac
