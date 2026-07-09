#!/usr/bin/env bash
# Sets up a fresh env with a torch build matching this cluster's GPU driver.
#
# Why: the H100 nodes' NVIDIA driver only supports up to CUDA 12.8, but a plain
# `pip install torch` currently resolves to a CUDA-13.0-linked build, which fails
# at runtime with "The NVIDIA driver on your system is too old" even though the
# driver itself is fine. Fix is to pin torch to the +cu128 wheel variant.
set -eo pipefail
source $HOME/.bashrc
ENV_NAME="${1:-cgrm}"

micromamba create -n "$ENV_NAME" python=3.12 -y
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# CUDA 12.8 nvcc + library headers, matching the driver's max supported CUDA
# version -- needed when deepspeed or flashinfer JIT-compiles ops at runtime.
# The -dev packages are required by flashinfer's sampling kernels (vllm builds
# them on first engine start): they need curand.h etc. and the libcuda.so stub.
micromamba install -c nvidia -c conda-forge \
    "cuda-nvcc=12.8" "cuda-libraries-dev=12.8" "cuda-cudart-dev=12.8" -y

# flashinfer's JIT linker hardcodes $CUDA_HOME/lib64{,/stubs}; conda envs only
# have lib/.
ln -sfn lib "$CONDA_PREFIX/lib64"

# JIT-built .so files are compiled with the conda toolchain's gcc/libstdc++,
# which is newer than the system /lib64/libstdc++.so.6 -- without this they
# fail to dlopen with "version GLIBCXX_3.4.32 not found".
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
printf '%s\n' 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"' \
    > "$CONDA_PREFIX/etc/conda/activate.d/prepend_lib.sh"

# deepspeed's op builder looks for $CUDA_HOME directly (not just nvcc on PATH)
# to detect the installed CUDA version -- without this it raises
# MissingCUDAException on import. Persist it on the env so it's set on every
# `micromamba activate`.
micromamba env config vars set -n "$ENV_NAME" CUDA_HOME="$CONDA_PREFIX"
micromamba activate "$ENV_NAME"

pip install "trl[wandb,deepspeed,vllm,liger]" hydra-core

# transformers 5.13.0 (2026-07-03) switched configs with mixed layer types
# (e.g. olmo3) to nested per-layer-type rope_parameters; vllm 0.23.0's olmo
# code still expects the flat format and dies with KeyError: 'rope_theta' at
# engine init. Pin to the last release before the change (same-day as vllm
# 0.23.0, so it's also what vllm was tested against).
pip install "transformers==5.12.1"

# Force the CUDA-12.8-linked torch build last so nothing pulled in above
# (vllm/deepspeed/trl deps) silently reinstalls a CUDA-13.0 build afterward.
# --force-reinstall is required: pip's "already satisfied" check ignores the
# +cu130/+cu128 local version suffix, so a plain install is a no-op here.
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"

# The default PyPI vllm wheel is compiled against CUDA 13 (its vllm._C links
# libcudart.so.13), which the cu128 torch stack above no longer provides --
# importing vllm then fails with "libcudart.so.13: cannot open shared object
# file". Swap in the +cu129 release wheel: it links libcudart.so.12, satisfied
# by torch's nvidia-cuda-runtime-cu12, and CUDA 12.x minor-version compat makes
# a 12.9-built binary fine on this 12.8 driver (verified via GPU smoke test).
# --no-deps so its metadata doesn't drag any CUDA-13 deps back in.
pip install --no-deps --force-reinstall \
    "https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# flash-attn has no prebuilt wheel for torch 2.11 yet. Rather than a slow
# from-source build, use the closest prebuilt wheel (built for torch 2.10,
# cu12, cp312, cxx11abiTRUE) -- verified working against torch 2.11.0+cu128
# on H100 via a live smoke test (flash_attn_func import + forward pass).
pip install --no-deps --force-reinstall \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

python -c "
import torch
print('torch:', torch.__version__, '| cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
# Catch CUDA-runtime linkage breakage at setup time, not at training launch.
# (Needs a GPU node: vllm._C also links libcuda.so.1 from the driver.)
from vllm import LLM, RequestOutput, SamplingParams
import vllm, flash_attn
print('vllm:', vllm.__version__, '| flash-attn:', flash_attn.__version__)
"
