# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test script for AiterSageFP8 attention backend.

This script tests two main scenarios:
1. Case 1: AiterSageFP8 forward produces valid (non-NaN) output
2. Case 2: Comparing AiterSageFP8 and SDPA backends for numerical closeness
"""

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.platforms import current_omni_platform

is_rocm = current_omni_platform.is_rocm()

try:
    from vllm_omni.diffusion.attention.backends.aiter_sage_fp8 import AiterSageFP8Impl

    HAS_AITER_SAGE_FP8 = True
except ImportError:
    HAS_AITER_SAGE_FP8 = False


@pytest.mark.skipif(not is_rocm, reason="AiterSageFP8 is only supported on ROCm")
@pytest.mark.skipif(not HAS_AITER_SAGE_FP8, reason="aiter with fav3_sage not available")
def test_aiter_sage_fp8_forward():
    """
    Case 1: Test that AiterSageFP8 forward produces valid output.

    - batch_size=1, seq_len=64, num_heads=8, head_dim=64
    - No attention mask
    - Verify output shape matches input and contains no NaN values
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 1
    seq_len = 64
    num_heads = 8
    head_dim = 64

    impl = AiterSageFP8Impl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=1.0 / (head_dim**0.5),
        causal=False,
    )

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    key = query.clone()
    value = query.clone()

    attn_metadata = AttentionMetadata(attn_mask=None)
    output = impl.forward(query=query, key=key, value=value, attn_metadata=attn_metadata)

    assert output.shape == query.shape, f"Output shape {output.shape} != input shape {query.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    print("\n=== Case 1: AiterSageFP8 Forward Test ===")
    print(f"Input shape:  {query.shape}")
    print(f"Output shape: {output.shape}")
    print("PASSED: Output is valid (no NaN/Inf)")


@pytest.mark.skipif(not is_rocm, reason="AiterSageFP8 is only supported on ROCm")
@pytest.mark.skipif(not HAS_AITER_SAGE_FP8, reason="aiter with fav3_sage not available")
def test_aiter_sage_fp8_vs_sdpa():
    """
    Case 2: Compare AiterSageFP8 and SDPA backends.

    - batch_size=2, seq_len=64, num_heads=8, head_dim=64
    - No attention mask
    - Compare outputs for numerical closeness

    Note: AiterSageFP8 uses FP8 quantization internally, so tolerances
    are higher than typical FP16/BF16 comparisons.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 2
    seq_len = 64
    num_heads = 8
    head_dim = 64

    sage_impl = AiterSageFP8Impl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=1.0 / (head_dim**0.5),
        causal=False,
    )

    sdpa_impl = SDPAImpl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=1.0 / (head_dim**0.5),
        causal=False,
    )

    torch.manual_seed(123)
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    key = query.clone()
    value = query.clone()

    attn_metadata = AttentionMetadata(attn_mask=None)

    output_sage = sage_impl.forward(
        query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata
    )
    output_sdpa = sdpa_impl.forward(
        query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata
    )

    max_diff = torch.max(torch.abs(output_sage - output_sdpa)).item()
    mean_diff = torch.mean(torch.abs(output_sage - output_sdpa)).item()

    print("\n=== Case 2: AiterSageFP8 vs SDPA Comparison ===")
    print(f"Batch size: {batch_size}")
    print(f"AiterSageFP8 output shape: {output_sage.shape}")
    print(f"SDPA output shape:         {output_sdpa.shape}")
    print(f"Max absolute difference:   {max_diff:.6f}")
    print(f"Mean absolute difference:  {mean_diff:.6f}")

    # Higher tolerance due to FP8 quantization in the sage kernel
    assert max_diff < 0.5, f"Max difference {max_diff} exceeds threshold 0.5"
    assert mean_diff < 0.05, f"Mean difference {mean_diff} exceeds threshold 0.05"

    print("PASSED: AiterSageFP8 and SDPA outputs are within tolerance!")


if __name__ == "__main__":
    print("Running AiterSageFP8 Attention Tests...")
    print("=" * 60)

    if not is_rocm:
        raise RuntimeError("These tests require a ROCm platform")
    if not HAS_AITER_SAGE_FP8:
        raise RuntimeError("aiter with fav3_sage support is required")

    try:
        print("\n[Running Case 1: AiterSageFP8 Forward]")
        test_aiter_sage_fp8_forward()
    except Exception as e:
        print(f"Case 1 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        print("\n[Running Case 2: AiterSageFP8 vs SDPA]")
        test_aiter_sage_fp8_vs_sdpa()
    except Exception as e:
        print(f"Case 2 failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test suite completed!")
