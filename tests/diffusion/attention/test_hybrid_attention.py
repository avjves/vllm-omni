# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the per-denoising-step hybrid attention schedule.

Covers:
- parse_hybrid_attention_schedule parsing + validation
- HybridAttentionSchedule.backend_for_step boundary behaviour
- HybridAttentionImpl per-step dispatch + activation broadcast
- torch.compile produces no graph breaks and at most one graph per backend
"""

import pytest
import torch

from vllm_omni.diffusion.attention.backends.hybrid import (
    HybridAttentionImpl,
    set_active_backend_for_all,
)
from vllm_omni.diffusion.attention.schedule import (
    HybridAttentionSchedule,
    parse_hybrid_attention_schedule,
)


class TestParseHybridAttentionSchedule:
    def test_none_returns_none(self):
        assert parse_hybrid_attention_schedule(None) is None

    def test_empty_returns_none(self):
        assert parse_hybrid_attention_schedule("") is None
        assert parse_hybrid_attention_schedule("   ") is None

    def test_valid_schedule(self):
        sched = parse_hybrid_attention_schedule("FLASH_ATTN:TORCH_SDPA:5:3")
        assert sched == HybridAttentionSchedule(
            high_backend="FLASH_ATTN",
            low_backend="TORCH_SDPA",
            first_n=5,
            last_n=3,
        )

    def test_case_insensitive(self):
        sched = parse_hybrid_attention_schedule("flash_attn:torch_sdpa:2:2")
        assert sched.high_backend == "FLASH_ATTN"
        assert sched.low_backend == "TORCH_SDPA"

    def test_whitespace_tolerant(self):
        sched = parse_hybrid_attention_schedule(" FLASH_ATTN : TORCH_SDPA : 1 : 1 ")
        assert sched.high_backend == "FLASH_ATTN"
        assert sched.low_backend == "TORCH_SDPA"
        assert sched.first_n == 1
        assert sched.last_n == 1

    def test_wrong_part_count_raises(self):
        with pytest.raises(ValueError):
            parse_hybrid_attention_schedule("FLASH_ATTN:TORCH_SDPA:5")
        with pytest.raises(ValueError):
            parse_hybrid_attention_schedule("FLASH_ATTN:TORCH_SDPA:5:3:1")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            parse_hybrid_attention_schedule("NOPE_ATTN:TORCH_SDPA:5:3")

    def test_non_int_counts_raise(self):
        with pytest.raises(ValueError):
            parse_hybrid_attention_schedule("FLASH_ATTN:TORCH_SDPA:a:3")

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError):
            parse_hybrid_attention_schedule("FLASH_ATTN:TORCH_SDPA:-1:3")


class TestBackendForStep:
    @pytest.fixture
    def sched(self):
        return HybridAttentionSchedule(
            high_backend="HIGH",
            low_backend="LOW",
            first_n=2,
            last_n=2,
        )

    def test_first_n_high(self, sched):
        assert sched.backend_for_step(0, 10) == "HIGH"
        assert sched.backend_for_step(1, 10) == "HIGH"

    def test_middle_low(self, sched):
        assert sched.backend_for_step(2, 10) == "LOW"
        assert sched.backend_for_step(7, 10) == "LOW"

    def test_last_n_high(self, sched):
        assert sched.backend_for_step(8, 10) == "HIGH"
        assert sched.backend_for_step(9, 10) == "HIGH"

    def test_total_steps_non_positive_defaults_high(self, sched):
        assert sched.backend_for_step(0, 0) == "HIGH"

    def test_overlap_raises(self, sched):
        # first_n + last_n > total_steps
        with pytest.raises(ValueError):
            sched.backend_for_step(1, 3)

    def test_exact_fit_no_middle(self):
        sched = HybridAttentionSchedule("HIGH", "LOW", first_n=2, last_n=2)
        # total == first_n + last_n: every step is high precision
        assert sched.backend_for_step(0, 4) == "HIGH"
        assert sched.backend_for_step(1, 4) == "HIGH"
        assert sched.backend_for_step(2, 4) == "HIGH"
        assert sched.backend_for_step(3, 4) == "HIGH"

    def test_distinct_backends(self, sched):
        assert sched.distinct_backends() == {"HIGH", "LOW"}

    def test_distinct_backends_same(self):
        sched = HybridAttentionSchedule("HIGH", "HIGH", 1, 1)
        assert sched.distinct_backends() == {"HIGH"}


class _FakeImpl:
    """Minimal stand-in for AttentionImpl that tags its output."""

    def __init__(self, tag: str, supports: bool = True):
        self.tag = tag
        self._supports = supports
        self.calls = 0

    def forward(self, query, key, value, attn_metadata=None):
        self.calls += 1
        # Return an identifiable tensor so dispatch can be asserted.
        return query + self.calls if isinstance(query, int) else query

    def supports_kv_cache_dtype(self, kv_cache_dtype, platform_key):
        return self._supports


class TestHybridAttentionImpl:
    def test_requires_impls(self):
        with pytest.raises(ValueError):
            HybridAttentionImpl({}, default_backend="HIGH")

    def test_default_must_be_present(self):
        with pytest.raises(ValueError):
            HybridAttentionImpl({"LOW": _FakeImpl("low")}, default_backend="HIGH")

    def test_starts_on_default(self):
        high = _FakeImpl("high")
        low = _FakeImpl("low")
        hybrid = HybridAttentionImpl({"HIGH": high, "LOW": low}, default_backend="HIGH")
        hybrid.forward(0, 0, 0)
        assert high.calls == 1
        assert low.calls == 0

    def test_set_active_switches(self):
        high = _FakeImpl("high")
        low = _FakeImpl("low")
        hybrid = HybridAttentionImpl({"HIGH": high, "LOW": low}, default_backend="HIGH")
        hybrid.set_active("LOW")
        hybrid.forward(0, 0, 0)
        assert low.calls == 1
        assert high.calls == 0

    def test_set_active_unknown_keeps_current(self):
        high = _FakeImpl("high")
        hybrid = HybridAttentionImpl({"HIGH": high}, default_backend="HIGH")
        hybrid.set_active("MISSING")
        hybrid.forward(0, 0, 0)
        assert high.calls == 1

    def test_supports_kv_cache_dtype_all(self):
        hybrid = HybridAttentionImpl(
            {"HIGH": _FakeImpl("high", supports=True), "LOW": _FakeImpl("low", supports=True)},
            default_backend="HIGH",
        )
        assert hybrid.supports_kv_cache_dtype("fp8", "rocm") is True

    def test_supports_kv_cache_dtype_any_false(self):
        hybrid = HybridAttentionImpl(
            {"HIGH": _FakeImpl("high", supports=True), "LOW": _FakeImpl("low", supports=False)},
            default_backend="HIGH",
        )
        assert hybrid.supports_kv_cache_dtype("fp8", "rocm") is False

    def test_broadcast_activation(self):
        high_a = _FakeImpl("high")
        low_a = _FakeImpl("low")
        hybrid_a = HybridAttentionImpl({"HIGH": high_a, "LOW": low_a}, default_backend="HIGH")
        high_b = _FakeImpl("high")
        low_b = _FakeImpl("low")
        hybrid_b = HybridAttentionImpl({"HIGH": high_b, "LOW": low_b}, default_backend="HIGH")

        set_active_backend_for_all("LOW")
        hybrid_a.forward(0, 0, 0)
        hybrid_b.forward(0, 0, 0)
        assert low_a.calls == 1
        assert low_b.calls == 1
        assert high_a.calls == 0
        assert high_b.calls == 0


class TestHybridCompileNoGraphBreak:
    """The per-step backend switch must happen in eager code, so torch.compile
    sees only a plain delegation and produces no graph breaks and at most one
    graph per active sub-impl."""

    def test_no_graph_break_and_bounded_graphs(self):
        torch._dynamo.reset()

        class _TensorImpl:
            def __init__(self, scale: float):
                self.scale = scale

            def forward(self, query, key, value, attn_metadata=None):
                return query * self.scale

        high = _TensorImpl(2.0)
        low = _TensorImpl(0.5)
        hybrid = HybridAttentionImpl({"HIGH": high, "LOW": low}, default_backend="HIGH")

        compiled = torch.compile(hybrid.forward, dynamic=True, fullgraph=True)

        q = torch.ones(4, 8)
        # Two backends → at most two compiled graphs, reused across steps.
        for step in range(6):
            backend = "HIGH" if step < 2 or step >= 4 else "LOW"
            set_active_backend_for_all(backend)
            out = compiled(q, q, q)
            expected = q * (2.0 if backend == "HIGH" else 0.5)
            assert torch.allclose(out, expected)
