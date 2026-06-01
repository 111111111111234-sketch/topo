# ============================================================================
# ConfTopo / ETPNav - 基于小鹏 fuyao 平台的 Dockerfile
# 基础镜像 25.05: CUDA 12.9 + PyTorch 2.8 + Python 3.12 (Blackwell GPU ready)
# ============================================================================
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/nvidia-pytorch:25.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=1
ENV TZ=Asia/Shanghai

ARG PIP_OPTIONS="-i https://nexus-wl.xiaopeng.link/repository/ai_infra_pypi_group/simple --timeout 600 --no-cache-dir"

# ----------------------------------------------------------------------------
# fuyao 平台标准组件
# 注意: nvidia-pytorch:25.05 为 Python 3.12, remote-kernel 要求 <=3.10
#       如遇 kernel 兼容问题可注释此段
# ----------------------------------------------------------------------------
COPY --from=infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/public:fuyao-base-1.9.7 \
    /opt/data-infra/remote-kernel /tmp/remote-kernel
RUN pip install ${PIP_OPTIONS} \
    /tmp/remote-kernel/jupyter_enterprise_gateway-3.2.2-py3-none-any.whl \
    /tmp/remote-kernel/jupyter_server-1.24.0-py3-none-any.whl \
    /tmp/remote-kernel/jupyterlab-3.6.6-py3-none-any.whl \
    jupyterlab-tensorboard-pro==3.0.0 \
    jupyterlab==3.6.6 \
    jupytext==1.16.1 \
    tensorboard==2.10.1 \
    tensorboardX==2.5 || true
RUN mkdir -p /usr/local/share/jupyter/ && \
    tar -C /usr/local/share/jupyter/ -xf /tmp/remote-kernel/jupyter-kernels.tar.gz || true
RUN rm -rf /tmp/remote-kernel

COPY --from=infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/public:fuyao-base-1.9.7 \
    /opt/data-infra /opt/data-infra
ENV PATH="${PATH}:/opt/data-infra"

# ----------------------------------------------------------------------------
# 系统依赖：habitat-sim EGL/OpenGL headless 渲染
# ----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libopengl0 \
      libegl1 \
      libgles2 \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      zsh \
      curl \
      wget \
      git \
      vim \
      zip; \
    rm -rf /var/lib/apt/lists/*

# NVIDIA EGL headless 渲染支持
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display

# ----------------------------------------------------------------------------
# fuyao CLI
# ----------------------------------------------------------------------------
RUN pip3 install fuyao \
    -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com \
    --extra-index-url http://nexus-wl.xiaopeng.link:8081/repository/ai_infra_pypi/simple --trusted-host nexus-wl.xiaopeng.link && \
    pip install --no-cache-dir --force-reinstall "packaging>=24.2"

# ----------------------------------------------------------------------------
# ConfTopo 主环境依赖 (直接用基础镜像 Python 3.12 + PyTorch 2.8)
# ----------------------------------------------------------------------------
RUN pip install --no-cache-dir ${PIP_OPTIONS} "setuptools<75" "packaging>=24.2"

RUN pip install --no-cache-dir ${PIP_OPTIONS} \
    numpy==1.26.4 scipy scikit-learn matplotlib==3.9.4 \
    pillow tqdm pyyaml networkx==3.2.1 \
    imageio imageio-ffmpeg h5py lmdb numba \
    opencv-python-headless \
    gym==0.26.2 hydra-core omegaconf yacs jsonlines \
    msgpack msgpack-numpy numpy-quaternion \
    transformers>=4.40 timm==1.0.27 \
    open_clip_torch==3.3.0 ftfy regex \
    openai safetensors huggingface_hub gdown requests \
    tensorboardX easydict natsort

# ----------------------------------------------------------------------------
# vLLM (共享基础 PyTorch 2.8)
# 使用官方 PyPI 作为备用源，避免 nexus 镜像 502
# ----------------------------------------------------------------------------
RUN pip install --no-cache-dir ${PIP_OPTIONS} \
    --extra-index-url https://pypi.org/simple \
    --retries 5 --timeout 300 \
    vllm

# ----------------------------------------------------------------------------
# Miniconda (goat 和 vlnce 需要不同 Python + habitat-sim 版本)
# ----------------------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash
ENV PATH="/opt/conda/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ----------------------------------------------------------------------------
# goat 环境: Python 3.9 + habitat-sim 0.2.4 + PyTorch 2.8
# ----------------------------------------------------------------------------
RUN conda create -n goat python=3.9 cmake "pip<24.4" -y

RUN conda install -n goat habitat-sim=0.2.4 headless -c aihabitat -c conda-forge -y

RUN conda run -n goat pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN conda install -n goat -c conda-forge --override-channels -y \
    numpy=1.26.4 scipy scikit-learn matplotlib pillow tqdm pyyaml

RUN conda run -n goat pip install --no-cache-dir "setuptools<70" && \
    conda run -n goat pip install --no-cache-dir ${PIP_OPTIONS} \
    opencv-python-headless \
    gym==0.23.0 hydra-core omegaconf yacs jsonlines \
    imageio imageio-ffmpeg msgpack msgpack-numpy h5py lmdb numba \
    transformers timm open_clip_torch ftfy regex \
    easydict tensorboardX gdown requests pytest-runner

# ----------------------------------------------------------------------------
# vlnce 环境: Python 3.8 + habitat-sim 0.1.7 + PyTorch 1.9.1
# (注: PyTorch 1.9.1 不支持 Blackwell GPU CUDA compute, 仅代码兼容)
# ETPNav 官方: numpy=1.19.5；habitat-sim 若单独装会拉到 numpy 1.26 导致 torch 1.9 冲突
# ----------------------------------------------------------------------------
RUN conda create -n vlnce python=3.8 "pip<24.4" -y

RUN conda install -n vlnce habitat-sim=0.1.7 headless \
    numpy=1.19.5 scipy=1.5.3 matplotlib=3.3.2 \
    -c aihabitat -c conda-forge -y

RUN conda run -n vlnce pip install --no-cache-dir --no-deps \
    torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN conda run -n vlnce pip install --no-cache-dir "setuptools<70"

# gym==0.21.0 在部分镜像源只有 sdist，会在新 setuptools 下 egg_info 失败；这里强制用 wheel
RUN env -u PIP_CONSTRAINT conda run -n vlnce pip install --no-cache-dir \
    --extra-index-url https://pypi.org/simple \
    --only-binary=:all: \
    gym==0.21.0

RUN env -u PIP_CONSTRAINT conda run -n vlnce pip install --no-cache-dir ${PIP_OPTIONS} \
    "numpy==1.19.5" \
    opencv-python-headless==4.5.5.64 \
    hydra-core omegaconf yacs jsonlines \
    msgpack msgpack-numpy h5py lmdb \
    transformers==4.12.5 tokenizers==0.10.3 timm==0.5.4 ftfy regex \
    networkx==2.8.8 easydict tensorboardX gdown requests

RUN env -u PIP_CONSTRAINT conda run -n vlnce pip install --no-cache-dir \
    git+https://github.com/openai/CLIP.git

# ----------------------------------------------------------------------------
# 终端美化 + zsh 插件
# ----------------------------------------------------------------------------
RUN RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

RUN git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

RUN sed -i 's/^ZSH_THEME=.*/ZSH_THEME="ys"/' /root/.zshrc && \
    sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting z extract)/' /root/.zshrc && \
    printf '\nexport TERM=xterm-256color\n' >> /root/.zshrc && \
    printf 'alias ls="ls --color=auto"\n' >> /root/.zshrc && \
    printf 'alias grep="grep --color=auto"\n' >> /root/.zshrc && \
    printf '\nsource /opt/conda/etc/profile.d/conda.sh\n' >> /root/.zshrc

ENV TERM=xterm-256color
ENV SHELL=/bin/zsh

# bash 进入时自动切换 zsh
RUN printf '\nif [ -n "$PS1" ] && [ -z "$ZSH_VERSION" ]; then\n  exec /bin/zsh -l\nfi\n' >> /root/.bashrc

# ----------------------------------------------------------------------------
# 启动
# 项目代码 + 模型 + 数据通过 volume 挂载
# ----------------------------------------------------------------------------
ENTRYPOINT ["tini", "-s", "--", "/bin/zsh", "-l"]
