# ETPNav + ConfTopo 环境依赖说明

> **说明**：本文档记录的是你在本项目中**曾经搭建并成功跑通**的环境配置（`vlnce39` / `conftopo` / `vllm_server` 等）。  
> 若当前机器上 conda 环境已被清除，本文档用于**复现**；复现命令见文末「一键复现」。

---

## 一、环境总览

| 环境名 | Python | 用途 |
|--------|--------|------|
| **`conftopo-dev`** | 3.12 | Core 开发 / 单测（`environment-dev.yaml`，**无** habitat / vLLM） |
| **`vllm`** | 3.10 | vLLM 推理服务（`environment-vllm.yaml`，Docker part1） |
| **`goat`** | 3.9 | **ConfTopo-GOAT** / GOAT-Bench（habitat **0.2.x**，torch **2.8+cu128**） |
| **`etpnav`** | 3.9 | **ConfTopo-ETPNav** / R2R-CE（habitat **0.1.7**，torch **2.8+cu128**） |
| **`vlnce`** | 3.8 | ETPNav 官方 baseline（torch 1.9，可选） |

**任务与环境对应：**

| 任务 | 环境 | 关键依赖 |
|------|------|----------|
| ConfTopo Core / 单测 | **`conftopo-dev`** | open_clip + networkx + `conftopo` 包 |
| LLM 服务 / GoalGraph 解析 API | **`vllm`** | vLLM + Qwen（`start-vllm`）；Agent 仅 `openai` 客户端 |
| GOAT-Bench / ConfTopo-GOAT | **`goat`** | habitat 0.2.x + torch **2.8+cu128** |
| R2R-CE / ConfTopo-ETPNav | **`etpnav`** | habitat 0.1.7 + torch **2.8+cu128** |
| ETPNav 官方复现 | **`vlnce`** | torch 1.9 + habitat 0.1.7 |

> **PyTorch**：主栈统一 `torch==2.8.0` + `cu128` index。Driver 显示 CUDA 13.0 可继续用 cu128；**勿**用 `2.8.0a0` / CUDA 13 替换 etpnav/goat 栈。

> GOAT 与 ETPNav 的 Habitat 版本不同（0.2.3 vs 0.1.7），**不要混装在同一 conda 环境**。

---

## 一点五、NGC 基础镜像 vs PyPI `cu128`（必读）

很多人会把两件事混在一起：**NVIDIA 没有「torch 2.8.0+cu128」的容器镜像**；`+cu128` 是 **PyPI 官方 wheel 标签**，用 `pip install` 装进各 conda 环境即可。

| 来源 | 是什么 | 本项目用法 |
|------|--------|------------|
| **`nvidia-pytorch:25.05-py3`**（Dockerfile.part1 `FROM`） | **CUDA 12.9.0** + **`2.8.0a0+5228986c39`** + TensorRT 10.10.0.31（NGC 官方组合） | **基础镜像栈**；fuyao / EGL / 编译 habitat 用其 CUDA toolkit；**goat/etpnav 的 torch 仍用 conda 内 PyPI cu128**（见下） |
| **PyPI `torch==2.8.0` + `cu128` index** | 稳定版 wheel（`2.8.0+cu128`） | **goat / etpnav / conftopo-dev** 里 `pip install`（与 ETPNav eval 对齐） |
| **PyPI `cu129` index** | 稳定版，CUDA 12.9 标签 | 与 NGC 25.05 **更接近**；换用需 **重跑 eval**，默认仍用 **cu128** |
| **`nvidia-pytorch:25.08+`** | **CUDA 13.0** + `2.8.0a0` | 仅实验；habitat-sim 0.1.7 / vLLM 0.19 未按此栈验证 |

**Driver 显示 CUDA 13.0** 只表示驱动上限；在 25.05（12.9）或 conda 里装 **cu128/cu129** 的 PyTorch 都可以跑，**不必**为了用 GPU 再去装 CUDA 13 toolkit 或换成 25.08 镜像。

**建议（Blackwell + 当前 Dockerfile）：**

1. **继续 `FROM nvidia-pytorch:25.05-py3`**（12.9 + 2.8.0a0），不要指望 NGC 提供 cu128 镜像。  
2. **goat / etpnav / dev** 继续在 conda 里装 **`torch==2.8.0` + `https://download.pytorch.org/whl/cu128`**（复现优先）。  
3. **vLLM** 单独 `vllm` 环境（依赖 CUDA 12.x 运行时）。  
4. **只有**你愿意重编 habitat、换 vLLM、重跑全流程时，才试 **25.08（CUDA 13）** 或 conda 里 **cu129**，并单独建实验环境，**不覆盖** etpnav/goat。

---

## 二、系统层（非 pip）

| 组件 | 版本/说明 |
|------|-----------|
| OS | Linux |
| GPU | NVIDIA RTX PRO 5000 72GB (Blackwell) |
| NVIDIA Driver | 580.126.09 |
| Docker 基础镜像 CUDA | **12.9**（NGC 25.05）；conda 内 PyTorch 标签 **cu128**（PyPI 稳定版） |
| 渲染 | EGL headless（habitat-sim 无显示器集群模式） |
| 编译 | cmake ≥ 3.14（habitat-sim 源码编译） |

---

## 三、ETPNav 官方环境 `vlnce`

定义文件：`ETPNav/environment.yaml`

### Conda 依赖

```
python=3.8
numpy=1.19.5, scipy=1.5.3, matplotlib=3.3.2
numba=0.53.1, llvmlite=0.36.0, pillow=8.3.2
imageio, imageio-ffmpeg, ffmpeg
ipython, ipykernel, jupyter_client
quaternion, tornado, tqdm, six, readline
setuptools, pip
habitat-sim=0.1.7 (headless, conda)
```

### Pip 依赖（environment.yaml）

```
bresenham, dtw, easydict, fastdtw, ftfy, future, gdown, gpustat
gym==0.21.0
h5py, higher, hydra-core, imagecorruptions, jsonlines, lmdb
moviepy, msgpack, msgpack-numpy, networkx
nvidia-ml-py3, omegaconf, opencv-python, progressbar2
protobuf, psutil, py360convert, pyyaml, regex, requests
sacremoses, scikit-image==0.17.2, scikit-learn==0.24.2
sentencepiece, shapely, stanza
tensorboardx, timm==0.5.4, tokenizers==0.10.3
transformers==4.12.5, webdataset, yacs
```

### README 额外安装

```
torch==1.9.1+cu111, torchvision==0.10.1+cu111
git+https://github.com/openai/CLIP.git
gym==0.21.0
habitat-lab v0.1.7 (源码 pip install -e)
```

---

## 四、升级后的 ETPNav + ConfTopo 栈（`vlnce39` / `conftopo`）

这是你实际跑通 R2R-CE eval、ConfTopo Phase 1 时使用的环境。

### 核心版本

| 包 | 版本 |
|----|------|
| Python | **3.9.23** |
| PyTorch | **2.8.0+cu128**（Blackwell GPU 必须 2.8+） |
| torchvision | 0.23.0 |
| torchaudio | 2.8.0 |
| numpy | **1.26.4** |
| habitat-sim | **0.1.7**（源码编译） |
| habitat-lab | **0.1.7**（editable 安装） |
| gym | **0.23.1**（原 0.21.0 与 py3.9 不兼容） |
| transformers | 4.12.5 |
| timm | 1.0.27 |
| networkx | 3.2.1 |
| matplotlib | 3.9.4 |
| opencv-python | 4.11.0.86 |

### ConfTopo 额外依赖

```
conftopo (pip install -e .)
open_clip_torch==3.3.0
openai==2.38.0
ftfy, regex, scipy
safetensors, huggingface_hub
gdown
```

### 源码 / editable 安装

```
habitat-sim 0.1.7     # 从源码编译
habitat-lab v0.1.7    # pip install -e habitat-lab/
conftopo              # pip install -e . (项目根 setup.py)
clip                  # git+openai/CLIP (可选)
```

### 已处理的兼容性问题

- `np.float` → `float`
- `gym.spaces.Discrete(0)` → `Discrete(1)`
- `torch.load` 默认 `weights_only=True` → `run.py` 中改为 `False`
- EGL 库与 NVIDIA 驱动版本对齐
- `attrs`、`numpy-quaternion` 与 numpy 版本冲突等

---

## 五、LLM 环境 `vllm_server`

| 包 | 版本 |
|----|------|
| vLLM | **0.19.1**（CUDA 12，`libcudart.so.12`） |
| 模型 | Qwen2.5-7B-Instruct @ `/root/workspace/models/Qwen2.5-7B-Instruct` |
| 启动脚本 | `ETPNav/scripts/start_vllm.sh` |

> PyPI 上 vLLM 0.21+ 需要 CUDA 13，与当前 PyTorch/CUDA 12.8 不兼容，请固定 **0.19.1**。

---

## 六、GOAT 环境 `goat`

| 包 | 版本 |
|----|------|
| Python | 3.8 |
| habitat-sim | **0.2.3** |
| habitat-lab | **0.2.3** @ `habitat-lab-v0.2.3/` |
| gym | 0.23.1 |
| hydra-core | 1.3.2 |

---

## 七、数据与模型路径

| 资源 | 路径 |
|------|------|
| MP3D 场景 (90) | `ETPNav/data/scene_datasets/mp3d/` |
| R2R-CE 数据 | `ETPNav/data/datasets/R2R_VLNCE_v1-2*/` |
| ETPNav 预训练权重 | `ETPNav/data/pretrained/` |
| CLIP ViT-B-32 | `/root/workspace/models/clip/` |
| Qwen2.5-7B | `/root/workspace/models/Qwen2.5-7B-Instruct/` |
| GOAT episodes | `models/goat-bench/data/datasets/goat_bench/hm3d/v1/` |
| HM3D val 场景 | `models/goat-bench/data/scene_datasets/hm3d/val/` |
| GoalGraphs | `data/goal_graphs/{r2r,goat}/` |

---

## 八、一键复现

### conftopo-dev + vllm

```bash
bash docker_build/install-part1-conda.sh
conda activate conftopo-dev && pip install -e . && python conftopo/tests/test_core.py
# 另开终端: start-vllm
```

### etpnav（ETPNav 0.1.7 栈）

`environment.yaml`（`name: etpnav`）+ Dockerfile.part5（habitat-sim 源码编译）。

```bash
# 1. 创建 conda 环境
conda env create -f environment.yaml
conda activate etpnav

# 2. habitat-sim 0.1.7 源码编译（ETPNav）
#    见 https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md

# 3. habitat-lab v0.1.7
cd habitat-lab
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt   # 注释掉 tensorflow 行
pip install -e .

# 4. ConfTopo + 项目
cd /path/to/project
pip install -e .

# 5. （可选）LLM 服务
conda create -n vllm_server python=3.10 -y
conda activate vllm_server
pip install vllm==0.19.1 -i https://pypi.org/simple/
bash ETPNav/scripts/start_vllm.sh
```

### 验证安装

```bash
conda activate conftopo
python -c "
import torch, habitat_sim
from conftopo import ConfTopoConfig
print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())
print('habitat-sim', habitat_sim.__version__)
print('ConfTopo OK')
"
```

---

## 九、相关文件

| 文件 | 说明 |
|------|------|
| `environment-dev.yaml` | **conftopo-dev**（Core / 单测） |
| `environment-vllm.yaml` | **vllm**（LLM 服务） |
| `environment.yaml` | **etpnav**（ConfTopo-ETPNav） |
| `ETPNav/environment.yaml` | ETPNav 官方 vlnce 环境 |
| `setup.py` | conftopo 包 editable 安装 |
| `ETPNav/docs/PLAN.md` | ConfTopo-Agent 实施计划 |
| `models/goat-bench/DOWNLOAD_GUIDE.md` | GOAT / HM3D 数据下载 |

---

*最后更新：2026-06-01*
