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

**重要**：

1. 必须在**项目根目录**执行 `fuyao docker`。若在 `docker/` 子目录执行，构建上下文只有 ~7KB，`COPY setup.py` / `conftopo` 会报 `not found`（日志里 `transferring context: 2B`）。
2. 需要 git **upstream**（`git rev-parse @{u}`）。无公司 GitLab 时可一次性执行：
   ```bash
   git init --bare /tmp/conftopo-fuyao-origin.git
   git remote add origin /tmp/conftopo-fuyao-origin.git   # 若尚无 origin
   git push -u origin master
   ```
   或直接使用 `bash scripts/fuyao-docker-part1.sh`（脚本会自动配置本地 origin）。

```bash
cd /workspace/tangyx7@xiaopeng.com

# Stage 1: conftopo-dev（推荐脚本，自动 cd 到根目录并确认危险目录提示）
bash scripts/fuyao-docker-part1.sh

# 或手动（需输入 y 确认危险目录）
echo y | fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part1

# → 记录推送的 tag，写入 Dockerfile.part2 的 ARG BASE_IMAGE

# Stage 2–4 同样在项目根目录执行
echo y | fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part2
echo y | fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part3
echo y | fuyao docker --site=fuyao_hk --push --dockerfile=docker/Dockerfile.part4
```

打包体积参考：part1 应远大于 7KB（含 `conftopo/`、`setup.py`、`docker_build/` 等，由 `.fuyaoignore` 过滤大目录）。

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
