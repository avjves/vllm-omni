# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composite attention impl for per-step hybrid (high/low precision) schedules.

``HybridAttentionImpl`` wraps several concrete ``AttentionImpl`` instances (one
per backend referenced by the schedule) and delegates each forward to whichever
sub-impl is currently *active*.

Compile safety
--------------
The active sub-impl is selected in eager code (``set_active_backend_for_all``,
called from ``set_forward_context_denoise_step_idx`` before the compiled
transformer blocks run) and stored as ``self._active_impl``. ``forward`` is a
plain delegation with **no branching on the denoise step**, so torch.compile
guards on the identity of ``_active_impl`` only: a hybrid schedule with two
backends yields at most two compiled graphs (reused across all steps), with no
graph breaks and no per-step recompilation.
"""

from __future__ import annotations

import weakref

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

# Live hybrid impls, broadcast-activated each denoise step. WeakSet so that
# discarded models (e.g. across tests / reloads) do not leak.
_REGISTERED_HYBRID_IMPLS: weakref.WeakSet[HybridAttentionImpl] = weakref.WeakSet()


class HybridAttentionImpl(AttentionImpl):
    """Delegates to one of several sub-impls based on the active backend."""

    def __init__(self, impls: dict[str, AttentionImpl], default_backend: str) -> None:
        if not impls:
            raise ValueError("HybridAttentionImpl requires at least one sub-impl.")
        if default_backend not in impls:
            raise ValueError(f"default_backend {default_backend!r} not among sub-impls {sorted(impls)}.")
        self._impls = impls
        self._default_backend = default_backend
        # Start on the (high precision) default so quality is preserved until the
        # first per-step activation runs, or if activation never happens.
        self._active_impl: AttentionImpl = impls[default_backend]
        _REGISTERED_HYBRID_IMPLS.add(self)

    def set_active(self, backend: str) -> None:
        impl = self._impls.get(backend)
        if impl is None:
            # Backend not part of this schedule; keep current. Should not happen.
            logger.warning_once("Hybrid attention: unknown active backend %r; keeping current.", backend)
            return
        self._active_impl = impl

    @property
    def active_backend(self) -> str:
        """Name of the currently active backend (for logging/debugging)."""
        for name, impl in self._impls.items():
            if impl is self._active_impl:
                return name
        return "<unknown>"

    def supports_kv_cache_dtype(self, kv_cache_dtype: str | None, platform_key: str) -> bool:
        # Only enable KV-cache quant when every backend in the schedule supports
        # it, otherwise quant would silently break on the steps that switch.
        return all(impl.supports_kv_cache_dtype(kv_cache_dtype, platform_key) for impl in self._impls.values())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._active_impl.forward(query, key, value, attn_metadata)


def set_active_backend_for_all(backend: str) -> int:
    """Activate ``backend`` on every live hybrid attention impl (eager only).

    Returns the number of live hybrid impls that were activated.
    """
    count = 0
    for impl in _REGISTERED_HYBRID_IMPLS:
        impl.set_active(backend)
        count += 1
    return count
