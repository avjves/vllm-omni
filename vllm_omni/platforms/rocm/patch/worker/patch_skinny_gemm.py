# SPDX-License-Identifier: Apache-2.0

"""Disable ROCm skinny GEMM kernels (wvSplitKrc) which can fail on matrix
shapes common in diffusion transformers."""

import os

os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
