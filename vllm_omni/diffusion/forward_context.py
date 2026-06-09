from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.ir
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


@dataclass
class ForwardContext:
    """
    set forward context for diffusion models
    """

    vllm_config: VllmConfig | None = None
    omni_diffusion_config: OmniDiffusionConfig | None = None
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None
    split_text_embed_in_sp: bool = False
    denoise_step_idx: int | None = None
    # Total number of denoising steps for the current request. Used to resolve
    # per-step schedules (e.g. hybrid attention's last-N high-precision steps).
    total_denoise_steps: int | None = None
    # Per-request reference latent for img2img DiT models (e.g. Ming)
    ref_latent: torch.Tensor | None = None
    # whether to split the text embed in sequence parallel, if True, the text embed will be split in sequence parallel

    # Sequence Parallel padding support
    # When sequence length is not divisible by SP world size, padding is added
    # These values are used by SequenceParallelGatherHook to remove padding,
    # and by attention layers to create attention masks dynamically
    sp_padding_size: int = 0
    # Original sequence length before padding (for removing padding in gather)
    sp_original_seq_len: int | None = None

    # Set by registry when _sp_plan hooks are applied.
    # When True, sp_active is determined by _sp_shard_depth (for _sp_plan hooks)
    # When False, sp_active defaults to True when sequence_parallel_size > 1 (for manual SP, standalone tests, etc.)
    sp_plan_hooks_applied: bool = False
    # SP active scope tracking within the _sp_plan hook mechanism.
    # Tracks the depth of SP sharding - incremented on shard, decremented on gather
    # Used by attention layers to determine if SP communication should be enabled
    _sp_shard_depth: int = 0

    @property
    def sp_active(self) -> bool:
        """Returns True when SP attention parallelism should be enabled.

        - If _sp_plan hooks are applied: use _sp_shard_depth (0 = outside sharded region).
        - If _sp_plan hooks are NOT applied: default to True when sequence_parallel_size > 1,
          since _sp_shard_depth is only meaningful within the _sp_plan hook mechanism.
        """
        if self.sp_plan_hooks_applied:
            return self._sp_shard_depth > 0
        # No _sp_plan: assume SP active when configured (manual SP, standalone tests)
        if self.omni_diffusion_config is None:
            raise ValueError(
                "omni_diffusion_config is not set when checking sp_active! "
                "This usually means set_forward_context() was not called. "
                "Please call with set_forward_context(omni_diffusion_config=...)."
            )

        sp_size = self.omni_diffusion_config.parallel_config.sequence_parallel_size
        return sp_size is not None and sp_size > 1

    def __post_init__(self):
        pass


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def get_ulysses_mode(*, default: str = "strict") -> str:
    """Resolve the Ulysses-SP mode from the current ForwardContext.

    Returns `default` when ForwardContext is unavailable or the diffusion
    config is not set.
    """
    if not is_forward_context_available():
        return default

    cfg = get_forward_context().omni_diffusion_config
    if cfg is None:
        return default

    parallel_config = cfg.parallel_config
    return str(getattr(parallel_config, "ulysses_mode", default))


def create_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    split_text_embed_in_sp: bool = False,
    denoise_step_idx: int | None = None,
):
    return ForwardContext(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        split_text_embed_in_sp=split_text_embed_in_sp,
        denoise_step_idx=denoise_step_idx,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    split_text_embed_in_sp: bool = False,
    denoise_step_idx: int | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, split_text_embed_in_sp, etc.
    Here we can inject common logic for every model forward pass.
    """
    forward_context = create_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        split_text_embed_in_sp=split_text_embed_in_sp,
        denoise_step_idx=denoise_step_idx,
    )
    # vLLM CustomOp dispatch (e.g. QKVParallelLinear) requires a global
    # vLLM config set via set_current_vllm_config().
    # Also set priority for vLLM IR ops (e.g. RMSNorm), copied from vllm/forward_context.py
    with override_forward_context(forward_context):
        if vllm_config is None:
            yield
        else:
            # Local import to avoid importing vllm.config.vllm at module import time.
            from vllm.config.vllm import set_current_vllm_config

            with (
                set_current_vllm_config(vllm_config),
                vllm_config.kernel_config.ir_op_priority.set_priority(),
                vllm.ir.enable_torch_wrap(vllm_config.compilation_config.ir_enable_torch_wrap),
            ):
                yield


def set_forward_context_denoise_step_idx(step_idx: int | None, total_steps: int | None = None) -> None:
    """Set the current diffusion denoise step on the active ForwardContext.

    When a hybrid attention schedule is configured, this also activates the
    backend for ``step_idx`` on all hybrid attention impls. This runs in eager
    code (before the compiled transformer blocks execute for the step), so the
    per-step backend switch never causes a graph break.

    ``total_steps`` should be passed by the denoise loop (e.g. ``len(timesteps)``)
    so the schedule can resolve its last-N high-precision steps. When omitted,
    the value previously stored on the context is reused.
    """
    if _forward_context is None:
        return
    _forward_context.denoise_step_idx = step_idx
    if total_steps is not None:
        _forward_context.total_denoise_steps = total_steps

    _maybe_activate_hybrid_attention(step_idx, _forward_context.total_denoise_steps)


def set_forward_context_total_denoise_steps(total_steps: int | None) -> None:
    """Set the total denoise step count on the active ForwardContext."""
    if _forward_context is not None:
        _forward_context.total_denoise_steps = total_steps


def _maybe_activate_hybrid_attention(step_idx: int | None, total_steps: int | None) -> None:
    if step_idx is None or _forward_context is None:
        return
    config = _forward_context.omni_diffusion_config
    schedule = getattr(config, "hybrid_attention_schedule", None) if config is not None else None
    if schedule is None:
        return
    if total_steps is None:
        # Without the total we cannot resolve last-N steps; keep the high-precision
        # default that hybrid impls start on.
        logger.warning_once(
            "[hybrid-attn] denoise step %s reached but total_steps is unknown; "
            "staying on high-precision default backend %r. The denoise loop must pass "
            "total_steps to set_forward_context_denoise_step_idx() for the schedule to engage.",
            step_idx,
            schedule.high_backend,
        )
        return
    # Local import to avoid a module-level import cycle.
    from vllm_omni.diffusion.attention.backends.hybrid import set_active_backend_for_all

    backend = schedule.backend_for_step(step_idx, total_steps)
    set_active_backend_for_all(backend)


def set_forward_context_ref_latent(ref_latent: torch.Tensor | None) -> None:
    """Set the per-request reference latent on the active ForwardContext.

    Used by img2img-capable DiT models (e.g. Ming-flash-omni-2.0) so the
    transformer can read the reference latent from request scope instead of
    module instance state.
    """
    if _forward_context is not None:
        _forward_context.ref_latent = ref_latent
