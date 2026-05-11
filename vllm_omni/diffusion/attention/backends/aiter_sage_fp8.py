# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
except ImportError:
    logger.warning(
        "AiterSageFP8Backend is not available. "
        "The `aiter` library with fav3_sage support is required. "
        "This backend is only supported on ROCm (gfx942/gfx950)."
    )
    raise ImportError


class AiterSageFP8Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [16, 32, 64, 96, 128, 160, 192, 224, 256, 512]

    @staticmethod
    def get_name() -> str:
        return "AITER_SAGE_FP8"

    @staticmethod
    def get_impl_cls() -> type["AiterSageFP8Impl"]:
        return AiterSageFP8Impl


class AiterSageFP8Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # vLLM Omni tensors are NHD (batch, seq, heads, dim) = "bshd"
        output = fav3_sage_wrapper_func(query, key, value, layout="bshd")
        return output
