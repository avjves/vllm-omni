# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

if current_omni_platform.is_rocm():
    try:
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
    except ImportError:
        logger.warning("AITER Sage Attention backend is not available. Please update AITER version.")
        pass
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.warning(
            "SageAttentionBackend is not available. You may install sage-attention"
            " by pip install git+https://github.com/thu-ml/SageAttention.git"
        )
        raise ImportError

# TODO add sage3 attention backend


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = fav3_sage_wrapper_func(
            query,
            key,
            value,
        )
        return output
