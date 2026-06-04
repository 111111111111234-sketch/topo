# ============================================================================
# ConfTopo / ETPNav - Optimized Stage 1: base environments
#
# Builds the reusable base image with fuyao, system/runtime dependencies,
# Miniconda, and the four final runtime environments:
#   - conftopo-main: project Python environment
#   - vlnce: VLNCE / ETP environment, habitat-sim is compiled in 2.dockerfile
#   - goat: GOAT environment with habitat-sim 0.2.4
#   - vllm: isolated LLM serving environment
#
# Example:
#   docker build -f 1.dockerfile -t conftopo-base:optimized .
# ============================================================================
FROM infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/nvidia-pytorch:25.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=1
ENV TZ=Asia/Shanghai
ENV PATH="/opt/conda/bin:${PATH}:/opt/data-infra"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

ARG PIP_OPTIONS="-i https://nexus-wl.xiaopeng.link/repository/ai_infra_pypi_group/simple --timeout 600 --no-cache-dir"

# ----------------------------------------------------------------------------
# fuyao platform components
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

# ----------------------------------------------------------------------------
# System dependencies: EGL/OpenGL, build tools, shell, and source control
# ----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      ffmpeg \
      git \
      libegl1 \
      libegl1-mesa-dev \
      libgl1 \
      libgl1-mesa-dev \
      libgles2 \
      libglib2.0-0 \
      libglm-dev \
      libjpeg-dev \
      libopengl0 \
      libsm6 \
      libxcursor-dev \
      libxext6 \
      libxi-dev \
      libxinerama-dev \
      libxrandr-dev \
      libxrender1 \
      ninja-build \
      vim \
      wget \
      zip \
      zsh; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ----------------------------------------------------------------------------
# fuyao CLI
# ----------------------------------------------------------------------------
RUN pip3 install fuyao \
      -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com \
      --extra-index-url http://nexus-wl.xiaopeng.link:8081/repository/ai_infra_pypi/simple --trusted-host nexus-wl.xiaopeng.link && \
    pip install --no-cache-dir --force-reinstall "packaging>=24.2"

# ----------------------------------------------------------------------------
# Miniconda
# ----------------------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ----------------------------------------------------------------------------
# conftopo-main: project runtime, no vLLM server packages
# ----------------------------------------------------------------------------
RUN conda create -n conftopo-main python=3.12 pip -y && \
    conda run -n conftopo-main pip install --no-cache-dir \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n conftopo-main pip install --no-cache-dir ${PIP_OPTIONS} \
      "setuptools<75" "packaging>=24.2" \
      numpy==1.26.4 scipy scikit-learn matplotlib==3.9.4 \
      pillow tqdm pyyaml networkx==3.2.1 \
      imageio imageio-ffmpeg h5py lmdb numba \
      opencv-python-headless \
      gym==0.26.2 hydra-core omegaconf yacs jsonlines \
      msgpack msgpack-numpy numpy-quaternion \
      "transformers>=4.40" timm==1.0.27 \
      open_clip_torch==3.3.0 ftfy regex \
      openai safetensors huggingface_hub gdown requests \
      tensorboardX easydict natsort

# ----------------------------------------------------------------------------
# goat: Python 3.9 + habitat-sim 0.2.4 + PyTorch 2.8
# ----------------------------------------------------------------------------
RUN conda create -n goat python=3.9 cmake "pip<24.4" -y && \
    conda install -n goat habitat-sim=0.2.4 headless -c aihabitat -c conda-forge -y && \
    conda run -n goat pip install --no-cache-dir \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    conda install -n goat -c conda-forge --override-channels -y \
      numpy=1.26.4 scipy scikit-learn matplotlib pillow tqdm pyyaml && \
    conda run -n goat pip install --no-cache-dir ${PIP_OPTIONS} \
      "setuptools<70" \
      opencv-python-headless \
      gym==0.23.0 hydra-core omegaconf yacs jsonlines \
      imageio imageio-ffmpeg msgpack msgpack-numpy h5py lmdb numba \
      transformers timm open_clip_torch ftfy regex \
      easydict tensorboardX gdown requests pytest-runner

# ----------------------------------------------------------------------------
# vlnce: modern Python 3.9 + PyTorch 2.8; habitat-sim 0.1.7 is built later
# ----------------------------------------------------------------------------
RUN conda create -n vlnce python=3.9 cmake "pip>=23,<24.4" -y && \
    conda run -n vlnce pip install --no-cache-dir \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n vlnce pip install --no-cache-dir ${PIP_OPTIONS} \
      numpy==1.26.4 scipy scikit-learn matplotlib==3.9.4 \
      opencv-python==4.11.0.86 networkx==3.2.1 pillow tqdm pyyaml \
      gym==0.23.1 hydra-core omegaconf yacs jsonlines \
      imageio imageio-ffmpeg quaternion \
      h5py lmdb msgpack msgpack-numpy numba numpy-quaternion \
      transformers==4.12.5 tokenizers==0.10.3 timm==1.0.27 \
      tensorboardX easydict ftfy regex \
      open_clip_torch==3.3.0 openai safetensors huggingface_hub gdown requests \
      scikit-image sentencepiece sacremoses stanza shapely \
      bresenham dtw fastdtw progressbar2 gpustat \
      "protobuf>=3.20,<5" webdataset attrs && \
    conda run -n vlnce pip install --no-cache-dir \
      git+https://github.com/openai/CLIP.git || true

# ----------------------------------------------------------------------------
# vllm: isolated LLM serving environment
# ----------------------------------------------------------------------------
RUN conda create -n vllm python=3.12 pip -y && \
    conda run -n vllm pip install --no-cache-dir ${PIP_OPTIONS} \
      --extra-index-url https://pypi.org/simple \
      --retries 5 --timeout 300 \
      vllm

WORKDIR /workspace/tangyx7@xiaopeng.com

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/bin/bash"]
