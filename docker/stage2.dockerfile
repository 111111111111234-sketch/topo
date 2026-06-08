# ============================================================================
# ConfTopo / ETPNav - Optimized Stage 2: final source/runtime image
#
# Builds on stage1 output. This pure dependency image compiles habitat-sim
# 0.1.7 for vlnce, installs project runtime dependencies, and writes the final
# runtime shell/profile integration. Project code, GOAT-Bench code, and data are
# expected to be mounted at runtime.
#
# Example:
#   docker build -f docker1/stage2.dockerfile \
#     --build-arg BASE_IMAGE=conftopo-base:optimized \
#     -t conftopo-final:optimized .
# ============================================================================

ARG BASE_IMAGE=infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:tangyx7-260601-1947@sha256:13dffa483f14a81fd473b2453602c7dde05a20610216b12bd597c1bcc8bc6fe4
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="conftopo-optimized-final"
LABEL org.opencontainers.image.description="four-env final image: conftopo-main, vlnce, goat, vllm"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV WORKSPACE=/workspace/tangyx7@xiaopeng.com
ENV CONFTopo_MAIN_ENV=conftopo-main
ENV CONFTopo_MAIN_PYTHON=/opt/conda/envs/conftopo-main/bin/python
ENV MAGNUM_LOG=quiet
ENV GLOG_minloglevel=2
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV TERM=xterm-256color
ENV SHELL=/bin/zsh
ENV MAX_JOBS=1

ARG PIP_OPTIONS="-i https://nexus-wl.xiaopeng.link/repository/ai_infra_pypi_group/simple --timeout 600"
ARG HABITAT_SIM_TAG=v0.1.7
ARG HABITAT_SIM_GIT=https://github.com/facebookresearch/habitat-sim.git
ARG HABITAT_LAB_V023_GIT=https://github.com/facebookresearch/habitat-lab.git
ARG HABITAT_LAB_V023_REF=v0.2.3

# GOAT needs Habitat-Lab 0.2.3 packages. Clone early so source presence is
# independent from local build-context filtering/tracking.
RUN set -eux; \
  habitat_v023_clone_ok=0; \
  for attempt in 1 2 3 4 5; do \
    rm -rf /tmp/habitat-lab-v023-src; \
    if git -c http.version=HTTP/1.1 -c http.postBuffer=1048576000 \
      clone --depth 1 --branch ${HABITAT_LAB_V023_REF} ${HABITAT_LAB_V023_GIT} /tmp/habitat-lab-v023-src; then \
      habitat_v023_clone_ok=1; \
      break; \
    fi; \
    echo "habitat-lab v0.2.3 clone failed (attempt ${attempt}/5), retrying..."; \
    sleep $((attempt * 8)); \
  done; \
  if [ "$habitat_v023_clone_ok" -ne 1 ]; then \
    echo "FATAL: failed to clone habitat-lab v0.2.3 after retries"; \
    exit 1; \
  fi; \
  mkdir -p /opt/src; \
  cp -a /tmp/habitat-lab-v023-src/habitat-lab /opt/src/habitat-lab-v023; \
  cp -a /tmp/habitat-lab-v023-src/habitat-baselines /opt/src/habitat-baselines-v023; \
  rm -rf /tmp/habitat-lab-v023-src

# Fail fast if build context filtering removed required habitat-baselines sources.
RUN set -eux; \
  if [ -d /opt/src/habitat-lab-v023/habitat-lab/habitat ]; then \
    mv /opt/src/habitat-lab-v023/habitat-lab/* /opt/src/habitat-lab-v023/; \
    rmdir /opt/src/habitat-lab-v023/habitat-lab; \
  fi; \
  if [ -d /opt/src/habitat-baselines-v023/habitat-baselines/habitat_baselines ]; then \
    mv /opt/src/habitat-baselines-v023/habitat-baselines/* /opt/src/habitat-baselines-v023/; \
    rmdir /opt/src/habitat-baselines-v023/habitat-baselines; \
  fi; \
  test -d /opt/src/habitat-baselines-v023/habitat_baselines/il/data; \
  test -d /opt/src/habitat-baselines-v023/habitat_baselines/rl/models; \
  ls -la /opt/src/habitat-baselines-v023/habitat_baselines/il/data; \
  ls -la /opt/src/habitat-baselines-v023/habitat_baselines/rl/models

# ----------------------------------------------------------------------------
# habitat-sim 0.1.7 for vlnce
# ----------------------------------------------------------------------------
RUN git clone --depth 1 --branch ${HABITAT_SIM_TAG} ${HABITAT_SIM_GIT} /tmp/habitat-sim && \
    env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH \
      conda run -n vlnce pip install --no-cache-dir ${PIP_OPTIONS} -r /tmp/habitat-sim/requirements.txt && \
    git -C /tmp/habitat-sim submodule update --init --depth 1 src/deps/pybind11 && \
    python -c "from pathlib import Path; p = Path('/tmp/habitat-sim/src/deps/pybind11/include/pybind11/attr.h'); t = p.read_text(); p.write_text(t if '#include <cstdint>' in t else t.replace('#include \\\"cast.h\\\"\\n', '#include \\\"cast.h\\\"\\n#include <cstdint>\\n', 1))" && \
    cd /tmp/habitat-sim && \
    CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    conda run -n vlnce python setup.py install --headless --with-cuda && \
    python -c "from pathlib import Path; import shutil; ext_dir = Path('/opt/conda/envs/vlnce/lib/python3.9/site-packages/habitat_sim/_ext'); env_lib_dir = Path('/opt/conda/envs/vlnce/lib'); libs = ('libCorradePluginManager', 'libCorradeUtility', 'libCorradeTestSuite'); [((lambda versioned, soname, plain: (None if versioned is None else ((soname.exists() or soname.symlink_to(versioned.name)), (plain.exists() or plain.symlink_to(versioned.name)), shutil.copy2(versioned, env_lib_dir / versioned.name), shutil.copy2(versioned, env_lib_dir / f'{lib_name}.so.2'), shutil.copy2(versioned, env_lib_dir / f'{lib_name}.so'))))(next(ext_dir.glob(f'{lib_name}.so.2.*'), None), ext_dir / f'{lib_name}.so.2', ext_dir / f'{lib_name}.so')) for lib_name in libs]" && \
    rm -rf /tmp/habitat-sim

# ----------------------------------------------------------------------------
# Stable helper and project dependency layers.
# ----------------------------------------------------------------------------
RUN mkdir -p ${WORKSPACE}

RUN env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir ${PIP_OPTIONS} \
      protobuf==3.20.1 \
      tensorboard==2.8.0 \
      webdataset==0.1.40 \
      "moviepy>=1.0.1" \
      "faster-fifo>=1.4.2" && \
    env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir ${PIP_OPTIONS} \
      ifcfg yacs einops "gym>=0.22,<0.24" \
      ftfy regex tqdm GPUtil trimesh seaborn open3d timm transformers sentencepiece \
      scikit-learn hydra-core omegaconf jsonlines imageio imageio-ffmpeg networkx && \
    env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir ${PIP_OPTIONS} \
      openai

RUN env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir ${PIP_OPTIONS} \
      git+https://github.com/openai/CLIP.git

RUN env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/vlnce/bin/pip install --no-cache-dir ${PIP_OPTIONS} \
      git+https://github.com/openai/CLIP.git

RUN touch /opt/src/README.md \
          /opt/src/habitat-lab-v023/README.md \
          /opt/src/habitat-baselines-v023/README.md && \
    env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir \
      /opt/src/habitat-lab-v023 --no-deps && \
    env -u PIP_CONSTRAINT env -u LD_LIBRARY_PATH /opt/conda/envs/goat/bin/pip install --no-cache-dir \
      /opt/src/habitat-baselines-v023 --no-deps && \
    rm -rf /opt/src/habitat-lab-v023 /opt/src/habitat-baselines-v023

# ----------------------------------------------------------------------------
# Terminal polish: zsh, oh-my-zsh, plugins, profile, and shortcuts
# ----------------------------------------------------------------------------
RUN /opt/conda/bin/conda config --set auto_activate_base false && \
    /opt/conda/bin/conda init zsh && \
    if [ ! -d /root/.oh-my-zsh ]; then \
      RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended; \
    fi && \
    mkdir -p /root/.oh-my-zsh/custom/plugins && \
    if [ ! -d /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions ]; then \
      git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions \
        /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions; \
    fi && \
    if [ ! -d /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting ]; then \
      git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git \
        /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting; \
    fi && \
    sed -i 's/^ZSH_THEME=.*/ZSH_THEME="ys"/' /root/.zshrc && \
    sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting z extract)/' /root/.zshrc

# ----------------------------------------------------------------------------
# Runtime profile, conda LD_LIBRARY_PATH isolation, and shell aliases
# ----------------------------------------------------------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '%s\n' \
      '# ConfTopo runtime defaults' \
      'export WORKSPACE=/workspace/tangyx7@xiaopeng.com' \
      'export CONFTopo_MAIN_ENV=conftopo-main' \
      'export CONFTopo_MAIN_PYTHON=/opt/conda/envs/conftopo-main/bin/python' \
      'export MAGNUM_LOG=quiet' \
      'export GLOG_minloglevel=2' \
      'export NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display' \
      'export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json' \
      > /etc/profile.d/conftopo-env.sh && \
    for env_name in conftopo-main vlnce goat vllm; do \
      mkdir -p /opt/conda/envs/${env_name}/etc/conda/activate.d \
               /opt/conda/envs/${env_name}/etc/conda/deactivate.d; \
      if [ "${env_name}" = "vlnce" ]; then \
        printf '%s\n' \
          '#!/bin/bash' \
          'export _CONFTOPO_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"' \
          'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/habitat_sim/_ext:${_CONFTOPO_SAVED_LD_LIBRARY_PATH:-}"' \
          > /opt/conda/envs/${env_name}/etc/conda/activate.d/conftopo_ld_path.sh; \
      else \
        printf '%s\n' \
          '#!/bin/bash' \
          'export _CONFTOPO_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"' \
          'unset LD_LIBRARY_PATH' \
          > /opt/conda/envs/${env_name}/etc/conda/activate.d/conftopo_ld_path.sh; \
      fi; \
      printf '%s\n' \
        '#!/bin/bash' \
        'export LD_LIBRARY_PATH="${_CONFTOPO_SAVED_LD_LIBRARY_PATH:-}"' \
        'unset _CONFTOPO_SAVED_LD_LIBRARY_PATH' \
        > /opt/conda/envs/${env_name}/etc/conda/deactivate.d/restore_ld_path.sh; \
      chmod +x /opt/conda/envs/${env_name}/etc/conda/activate.d/conftopo_ld_path.sh \
               /opt/conda/envs/${env_name}/etc/conda/deactivate.d/restore_ld_path.sh; \
    done && \
    grep -q 'CONFTOPO_ENV_MARKER' /root/.zshrc 2>/dev/null || printf '%s\n' \
      '' \
      '# CONFTOPO_ENV_MARKER' \
      'export TERM=xterm-256color' \
      'source /etc/profile.d/conftopo-env.sh' \
      '[ -f /opt/conda/etc/profile.d/conda.sh ] && source /opt/conda/etc/profile.d/conda.sh' \
      'alias conftopo-main="conda activate conftopo-main"' \
      'alias conftopo-vlnce="conda activate vlnce"' \
      'alias conftopo-etp="conda activate vlnce"' \
      'alias conftopo-goat="conda activate goat"' \
      'alias vllm-env="conda activate vllm"' \
      'alias ls="ls --color=auto"' \
      'alias grep="grep --color=auto"' \
      >> /root/.zshrc && \
    printf '%s\n' \
      'if [ -n "$PS1" ] && [ -z "$ZSH_VERSION" ]; then' \
      '  exec /bin/zsh -l' \
      'fi' \
      >> /root/.bashrc

RUN cp /root/.zshrc /root/.zshrc.bak.$(date +%Y%m%d_%H%M%S) && \
    python -c 'from pathlib import Path; p = Path("/root/.zshrc"); s = p.read_text() if p.exists() else ""; block = "\n".join(["", "# >>> oh-my-zsh >>>", "export ZSH=\"$HOME/.oh-my-zsh\"", "", "ZSH_THEME=\"ys\"", "", "plugins=(", "  git", "  zsh-autosuggestions", "  zsh-syntax-highlighting", ")", "", "source \"$ZSH/oh-my-zsh.sh\"", "# <<< oh-my-zsh <<<", ""]) + "\n"; marker = "# CONFTOPO_ENV_MARKER"; p.write_text(s if "oh-my-zsh.sh" in s else (s.replace(marker, block + marker, 1) if marker in s else s + block))'

RUN /bin/zsh -lic 'source /root/.zshrc >/dev/null 2>&1 || true'

WORKDIR ${WORKSPACE}

ENTRYPOINT ["tini", "-s", "--", "/bin/zsh", "-l"]
