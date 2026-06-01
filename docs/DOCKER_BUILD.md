# ConfTopo Docker 分阶段构建指南

镜像分为 4 个阶段，每阶段创建 1 个 conda 环境。

**基础镜像（已选定）**：`nvidia-pytorch:25.05-py3`

| 组件 | 版本 |
|------|------|
| CUDA | **12.9.0** |
| PyTorch（NGC 系统 Python） | **2.8.0a0+5228986c39** |
| TensorRT | **10.10.0.31** |

各 conda 环境另装 PyPI **`torch==2.8.0+cu128`**（稳定版，与 ETPNav eval 对齐；NGC 无 cu128 同名镜像）。
TensorRT 仅部署推理时用，训练 / habitat 不依赖。

## 阶段总览

| 阶段 | 文件 | conda 环境 | 内容 |
|------|------|------------|------|
| **1** | `Dockerfile.part1` | **conftopo-dev** | NGC 基础 + fuyao + apt + Miniconda + Core 开发环境 + zsh + 源码 |
| **2** | `Dockerfile.part2` | **vllm** | LLM 推理（Py3.10 + vLLM） |
| **3** | `Dockerfile.part3` | **goat** | GOAT-Bench（Py3.9 + habitat-sim 0.2.4 + habitat-lab 0.2.3） |
| **4** | `Dockerfile.part4` | **etpnav** | ETPNav（Py3.9 + habitat-sim 0.1.7 源码 + habitat-lab 0.1.7） |

## 环境对应关系

| 用途 | conda 环境 | 激活 |
|------|------------|------|
| ConfTopo Core / 单测 | **conftopo-dev** | `conda activate conftopo-dev` |
| vLLM 推理服务 | **vllm** | `start-vllm`（`CONFTopo_VLLM_PYTHON`） |
| ConfTopo-GOAT / GOAT-Bench | **goat** | `conda activate goat` |
| ConfTopo-ETPNav / R2R-CE | **etpnav** | `conda activate etpnav` |

**ConfTopo-Agent**：GOAT 线在 `goat`，ETPNav 线在 `etpnav`；LLM 只在 `vllm` 进程，Agent 用 `openai` 调 API。

## 构建命令

```bash
cd /workspace/tangyx7@xiaopeng.com

# Stage 1: conftopo-dev
fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part1
# → 记录推送的 tag，写入 Dockerfile.part2 的 BASE_IMAGE

# Stage 2: vllm
fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part2

# Stage 3: goat
fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part3

# Stage 4: etpnav（habitat-sim 0.1.7 源码编译，耗时较长）
fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part4
```

每一阶段推送后，将镜像 tag 更新到下一阶段 `ARG BASE_IMAGE=...`。

## 容器内验证

```bash
verify-envs

conda activate conftopo-dev && python conftopo/tests/test_core.py
conda activate goat && python conftopo/tests/test_phase2.py
conda activate etpnav && python conftopo/tests/test_etpnav.py

# 终端 A
start-vllm 8000 0.25
# 终端 B: goat 或 etpnav 跑 Agent
```

## 运行时挂载

- 项目代码: `/workspace/tangyx7@xiaopeng.com`
- 模型: `models/Qwen3-4B`

```bash
conda activate conftopo-dev && pip install -e .
conda activate goat && pip install -e .
conda activate etpnav && pip install -e .
```

## 已知限制

- 激活 goat/etpnav/conftopo-dev/vllm 时会 `unset LD_LIBRARY_PATH`，避免基础镜像库污染 conda。
- **part4** habitat-sim 0.1.7 源码编译耗时较长。

## 环境声明文件

| 文件 | conda 名 |
|------|----------|
| `environment-dev.yaml` | conftopo-dev |
| `environment-vllm.yaml` | vllm |
| `environment.yaml` | etpnav |
