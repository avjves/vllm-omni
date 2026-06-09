# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-denoising-step hybrid attention schedule.

A hybrid schedule runs a *high precision* attention backend during the first
and last few denoising steps (which are the most quality-sensitive) and a
*low precision* backend for the steps in the middle, where quantization error
is tolerable. This trades a small amount of quality for faster middle steps.

The schedule is configured with a compact string::

    "<HIGH>:<LOW>:<first_n>:<last_n>"

e.g. ``"FLASH_ATTN:TORCH_SDPA:5:5"`` keeps FLASH_ATTN for the first 5 and last
5 steps and uses TORCH_SDPA in between. Backend names are the canonical
``DiffusionAttentionBackendEnum`` member names (case-insensitive).

Resolution (step -> backend) happens in eager code, never inside a compiled
region: see ``forward_context.set_forward_context_denoise_step_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class HybridAttentionSchedule:
    """High precision at the first/last N steps, low precision in the middle.

    ``high_backend`` and ``low_backend`` are canonical
    ``DiffusionAttentionBackendEnum`` member names (e.g. ``"FLASH_ATTN"``).
    """

    high_backend: str
    low_backend: str
    first_n: int
    last_n: int

    def distinct_backends(self) -> set[str]:
        return {self.high_backend, self.low_backend}

    def backend_for_step(self, step: int, total_steps: int) -> str:
        """Return the backend name to use at ``step`` of ``total_steps``.

        High precision when ``step`` falls in the first ``first_n`` or last
        ``last_n`` steps; low precision otherwise.
        """
        if total_steps <= 0:
            return self.high_backend
        if self.first_n + self.last_n > total_steps:
            raise ValueError(
                f"Hybrid attention schedule high-precision steps "
                f"(first_n={self.first_n} + last_n={self.last_n}) exceed "
                f"total denoising steps ({total_steps})."
            )
        if step < self.first_n or step >= total_steps - self.last_n:
            return self.high_backend
        return self.low_backend


def _resolve_backend_name(token: str) -> str:
    """Validate a backend token against the registry and return its canonical name."""
    name = token.strip().upper()
    if not name:
        raise ValueError("Empty backend name in hybrid attention schedule.")
    # Raises ValueError listing valid names if unknown (see registry metaclass).
    return DiffusionAttentionBackendEnum[name].name


def parse_hybrid_attention_schedule(spec: str | None) -> HybridAttentionSchedule | None:
    """Parse a ``"<HIGH>:<LOW>:<first_n>:<last_n>"`` string.

    Returns ``None`` when ``spec`` is ``None``/empty (feature disabled).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None

    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            "Hybrid attention schedule must be "
            "'<HIGH_PRECISION>:<LOW_PRECISION>:<first_n>:<last_n>', "
            f"e.g. 'FLASH_ATTN:TORCH_SDPA:5:5'. Got: {spec!r}"
        )

    high_backend = _resolve_backend_name(parts[0])
    low_backend = _resolve_backend_name(parts[1])
    try:
        first_n = int(parts[2])
        last_n = int(parts[3])
    except ValueError:
        raise ValueError(f"Hybrid attention schedule first_n/last_n must be integers. Got: {spec!r}") from None
    if first_n < 0 or last_n < 0:
        raise ValueError(f"Hybrid attention schedule first_n/last_n must be >= 0. Got: {spec!r}")

    schedule = HybridAttentionSchedule(
        high_backend=high_backend,
        low_backend=low_backend,
        first_n=first_n,
        last_n=last_n,
    )
    logger.info(
        "Hybrid attention schedule enabled: high=%s, low=%s, first_n=%d, last_n=%d",
        high_backend,
        low_backend,
        first_n,
        last_n,
    )
    return schedule
