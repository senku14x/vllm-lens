"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  What each request wants
(``extra_args["output_residual_stream"]``, its steering vectors and hooks)
is resolved once per request, and each forward pass is planned once from
the model runner's host-side buffers, so the hooks of a layer with nothing
to do return after one lookup and never synchronise with the device.
Passes whose plan has no hook work may replay CUDA graphs; passes with hook
work run eagerly (see ``_install_runner_wrappers``).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import pickle
import sys
import time
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers._hook_output import _replace_hook_output
from vllm_lens._helpers._serialize import pack_tensor
from vllm_lens._helpers.types import Hook, HookContext, SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_DTYPE_LIST = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.float64,
]
_DTYPE_TO_IDX_MAP = {d: i for i, d in enumerate(_DTYPE_LIST)}


def _dtype_to_idx(dtype: torch.dtype) -> int:
    return _DTYPE_TO_IDX_MAP.get(dtype, 0)


_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


_LAYER_PATH_ENV = "VLLM_LENS_LAYER_PATH"


class LayerDiscoveryError(RuntimeError):
    """vllm-lens could not locate the model's decoder layers."""


def _decoder_mixer_types() -> tuple[type, ...]:
    """Registry classes that live inside a decoder block.

    Only these are treated as decoder-layer markers.  Anything else that
    registers in ``static_forward_context`` (``FusedMoE``, multimodal
    encoder attention, ``CacheOnlyAttentionLayer`` KV-cache dummies,
    DeepSeek-V3.2 indexer caches, future unknowns) is ignored rather than
    guessed at, so a new vLLM layer type can't silently masquerade as a
    decoder layer.
    """
    types: list[type] = []
    from vllm.model_executor.layers.attention import Attention, MLAAttention

    types += [Attention, MLAAttention]
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        types.append(MambaBase)
    except ImportError:  # pragma: no cover - older vLLM without Mamba
        pass
    return tuple(types)


def _is_decoder_mixer(module: Any, mixer_types: tuple[type, ...]) -> bool:
    if not isinstance(module, mixer_types):
        return False
    # Encoder / cross-attention is not part of the decoder residual
    # stream.  Only ``Attention`` carries ``attn_type``; MLA and Mamba
    # mixers are decoder-only by construction.
    # ``attn_type`` is vLLM's ``AttentionType`` (a ``str`` Enum): compare by
    # value — ``str()`` of a str-Enum member is its *name* on Python ≥3.11.
    attn_type = getattr(module, "attn_type", None)
    return attn_type is None or attn_type == "decoder"


def _layers_from_registry(
    model: torch.nn.Module, registry: dict[str, Any]
) -> dict[int, torch.nn.Module]:
    """Map global decoder-layer index → module via ``static_forward_context``.

    Every attention-like layer registers there at construction under its
    full module prefix (e.g. ``model.layers.7.self_attn.attn``).  The
    decoder layer is the prefix truncated at its first integer segment,
    which is a *global* index even under pipeline parallelism (non-owned
    layers never construct, so they never register).
    """
    mixer_types = _decoder_mixer_types()
    layer_map: dict[int, torch.nn.Module] = {}
    for prefix, module in registry.items():
        if not _is_decoder_mixer(module, mixer_types):
            continue
        segments = prefix.split(".")
        idx_pos = next((i for i, seg in enumerate(segments) if seg.isdigit()), None)
        if idx_pos is None:
            raise LayerDiscoveryError(
                f"Registered layer {prefix!r} has no integer layer index in its "
                f"module path, so its decoder layer cannot be inferred."
            )
        layer_idx = int(segments[idx_pos])
        parent_path = ".".join(segments[: idx_pos + 1])
        try:
            layer = model.get_submodule(parent_path)
        except AttributeError as e:
            raise LayerDiscoveryError(
                f"Could not resolve decoder layer {parent_path!r} (from registered "
                f"layer {prefix!r}) on {type(model).__name__}."
            ) from e
        existing = layer_map.get(layer_idx)
        if existing is not None and existing is not layer:
            raise LayerDiscoveryError(
                f"Layer index {layer_idx} resolves to two different modules "
                f"({prefix!r} vs an earlier registry entry)."
            )
        layer_map[layer_idx] = layer
    return layer_map


def _layers_from_path_template(
    model: torch.nn.Module, template: str, total_layers: int
) -> dict[int, torch.nn.Module]:
    """Map layer index → module from a user-supplied path template.

    ``template`` is a dotted ``get_submodule`` path containing ``{i}``,
    e.g. ``"model.blocks.{i}"``.  Every index in ``range(total_layers)``
    must resolve; layers that resolve to ``PPMissingLayer`` belong to
    another pipeline stage and are skipped.
    """
    if "{i}" not in template:
        raise LayerDiscoveryError(
            f"{_LAYER_PATH_ENV}={template!r} must contain a '{{i}}' placeholder "
            f"for the layer index, e.g. 'model.layers.{{i}}'."
        )
    layer_map: dict[int, torch.nn.Module] = {}
    for i in range(total_layers):
        path = template.format(i=i)
        try:
            layer = model.get_submodule(path)
        except AttributeError as e:
            raise LayerDiscoveryError(
                f"{_LAYER_PATH_ENV}={template!r}: {path!r} does not exist on "
                f"{type(model).__name__}."
            ) from e
        if isinstance(layer, PPMissingLayer):
            continue
        layer_map[i] = layer
    return layer_map


def _runner_model(runner: Any) -> torch.nn.Module:
    """The model module itself, unwrapped from vLLM's CUDA-graph wrapper."""
    get_model = getattr(runner, "get_model", None)
    if callable(get_model):
        try:
            return get_model()
        except Exception:  # pragma: no cover - runner without a loaded model
            pass
    return runner.model


def _discover_layer_modules(worker: Any) -> dict[int, torch.nn.Module]:
    """Map global decoder-layer index → decoder-layer module on this rank.

    Resolution order:

    1. ``VLLM_LENS_LAYER_PATH`` (a ``get_submodule`` path template such as
       ``"model.layers.{i}"``), if set — used exclusively.
    2. vLLM's ``static_forward_context`` registry (see
       :func:`_layers_from_registry`).

    The result is checked against the config's ``num_hidden_layers`` —
    every index must be in range and this rank must own exactly the
    layers pipeline parallelism assigns it.  Any failure raises
    :class:`LayerDiscoveryError` naming the fix; there is deliberately no
    silent fallback, since a wrong layer map yields wrong activations.
    """
    model = _runner_model(worker.model_runner)
    total = _get_total_num_layers(worker)
    expected_local = int(worker.model_config.get_num_layers(worker.parallel_config))

    template = os.environ.get(_LAYER_PATH_ENV, "").strip()
    if template:
        layer_map = _layers_from_path_template(model, template, total)
        source = f"{_LAYER_PATH_ENV}={template!r}"
    else:
        registry = getattr(
            getattr(worker, "compilation_config", None), "static_forward_context", None
        )
        if not registry:
            raise LayerDiscoveryError(
                f"vLLM's static_forward_context registry is empty on "
                f"{type(model).__name__}; cannot discover decoder layers. "
                f"Set {_LAYER_PATH_ENV} (e.g. 'model.layers.{{i}}') to override."
            )
        layer_map = _layers_from_registry(model, registry)
        source = "static_forward_context registry"

    found = sorted(layer_map)
    out_of_range = [i for i in found if i < 0 or i >= total]
    if out_of_range or len(found) != expected_local:
        raise LayerDiscoveryError(
            f"Decoder-layer discovery via {source} found {len(found)} layer(s) "
            f"on this rank (indices {found}), but the model config declares "
            f"num_hidden_layers={total} with {expected_local} layer(s) on this "
            f"pipeline stage"
            + (f"; indices out of range: {out_of_range}" if out_of_range else "")
            + f". Set {_LAYER_PATH_ENV} to a get_submodule path template "
            f"(e.g. 'model.layers.{{i}}') to specify the decoder layers manually."
        )
    logger.info("vllm-lens: discovered %d decoder layer(s) via %s", len(found), source)
    return layer_map


def _get_total_num_layers(worker: Any) -> int:
    """Total decoder layers across all PP ranks (from the HF config)."""
    return int(worker.model_config.get_total_num_hidden_layers())


def _prefix_keys(internal_req_id: str) -> Iterator[str]:
    """Yield every ``k`` with ``internal_req_id.startswith(k + "-")``.

    vLLM turns an external request id into ``"{external_id}-{suffix}"``, so a
    registered key can only match if it ends right before a ``-`` of the
    internal id.  Enumerating those prefixes finds the same keys as testing
    every registered key, with a handful of dict lookups.
    """
    i = internal_req_id.find("-")
    while i != -1:
        yield internal_req_id[:i]
        i = internal_req_id.find("-", i + 1)


def _matching_keys(
    data: dict[str, Any],
    order: dict[str, int],
    internal_req_id: str,
    sentinel: str | None,
) -> list[str]:
    """Keys of ``data`` that apply to a request, in scan order.

    Same result as iterating ``data`` in insertion order and keeping each key
    for which ``internal_req_id.startswith(f"{key}-")`` (the async path), then
    appending the offline sentinel key from ``extra_args`` if registered.
    """
    if not data:
        return []
    found = [k for k in _prefix_keys(internal_req_id) if k in data]
    if len(found) > 1:
        found.sort(key=lambda k: order.get(k, 0))
    if sentinel and sentinel in data:
        found.append(sentinel)
    return found


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).
    """
    sentinel = extra_args.get("_steering_id") if extra_args else None
    keys = _matching_keys(
        extension._steering_data, extension._steering_order, internal_req_id, sentinel
    )
    return [cfg for k in keys for cfg in extension._steering_data[k]]


def _find_hook_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find all hook definitions that apply to an internal request ID.

    Checks three sources (in order):
    1. Per-request hooks keyed by external ID prefix (async path).
    2. Per-request hooks keyed by ``_hook_id`` sentinel (offline path).
    3. Persistent hooks (apply to every request).
    """
    results = _find_hook_configs_no_persistent(extension, internal_req_id, extra_args)
    results.extend(extension._persistent_hooks)
    return results


def _find_hook_configs_no_persistent(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find per-request hook definitions only (excludes persistent hooks)."""
    sentinel = extra_args.get("_hook_id") if extra_args else None
    keys = _matching_keys(
        extension._hook_data, extension._hook_order, internal_req_id, sentinel
    )
    return [h for k in keys for h in extension._hook_data[k]]


def _parse_layer_request(value: Any) -> tuple[bool, frozenset[int] | None]:
    """Normalise ``output_residual_stream`` / ``output_qk``.

    Returns ``(requested, layers)``; ``layers`` is ``None`` for "every layer".
    ``vllm_xargs`` passes values as JSON strings; a string that is not JSON
    is treated as truthy (capture all layers), as is any non-list value.
    """
    if value is None:
        return False, None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return True, None
    if isinstance(value, (list, tuple)):
        # Upstream tested ``layer_idx in value``; unhashable entries could
        # never equal a layer index, so dropping them keeps that meaning.
        return True, frozenset(x for x in value if isinstance(x, Hashable))
    return True, None


# ---------------------------------------------------------------------------
# Per-request resolution and per-pass plans
#
# Every decoder-layer hook fires on every forward pass.  Resolving what each
# request wants once (``_ReqPlan``, cached until the registered steering or
# hook data changes) and planning each pass once from the model runner's
# host-side buffers (``_StepPlan``) keeps the per-layer cost of a pass with
# nothing to do at one dict lookup, with no device syncs.
# ---------------------------------------------------------------------------

_NO_POS = 1 << 62
# Prune the per-request resolution cache once it holds this many entries
# beyond the live batch (entries of finished requests).
_REQ_CACHE_SLACK = 4096
# Log the first few hook errors with a traceback, then every N-th one.
_ERROR_LOG_FIRST = 5
_ERROR_LOG_EVERY = 1000


@dataclass(slots=True)
class _ReqPlan:
    """What one request wants from the hooks, resolved once and cached."""

    gen: int
    """``_lens_gen`` at resolution; a mismatch means the registered data changed."""
    sp_id: int
    """``id()`` of the request's SamplingParams (guards against a reused id)."""
    steer: list[SteeringVector]
    """Matching steering configs, flattened in application order."""
    steer_layers: frozenset[int]
    steer_broadcast: bool
    """Some config is 2-D: it applies to every position, generated ones too."""
    steer_min_pos: int
    steer_max_pos: int
    hooks: list[Hook]
    """Per-request hooks (persistent hooks are handled separately)."""
    pre_layers: frozenset[int]
    post_layers: frozenset[int]
    cap: bool
    cap_layers: frozenset[int] | None
    qk: bool
    qk_layers: frozenset[int] | None

    @property
    def idle(self) -> bool:
        return not (self.steer or self.hooks or self.cap or self.qk)


def _resolve_request(
    extension: HiddenStatesExtension, runner: Any, req_id: str, gen: int
) -> _ReqPlan | None:
    req_state = runner.requests.get(req_id)
    if req_state is None:
        return None
    sp = req_state.sampling_params
    extra = sp.extra_args if sp is not None else None
    configs = _find_steering_configs(extension, req_id, extra)
    layers: set[int] = set()
    broadcast = False
    min_pos, max_pos = _NO_POS, -1
    for cfg in configs:
        layers.update(cfg.layer_indices)
        if cfg.activations.dim() == 2:
            broadcast = True
            continue
        n_pos = int(cfg.activations.shape[1])
        # _apply_steering stops at pi >= n_pos, so only the first n_pos count.
        pos = (
            list(cfg.position_indices[:n_pos])
            if cfg.position_indices is not None
            else list(range(n_pos))
        )
        if pos:
            min_pos, max_pos = min(min_pos, min(pos)), max(max_pos, max(pos))
    hooks = _find_hook_configs_no_persistent(extension, req_id, extra)
    pre_layers = frozenset(i for h in hooks if h.pre for i in h.layer_indices)
    post_layers = frozenset(i for h in hooks if not h.pre for i in h.layer_indices)
    cap, cap_layers = _parse_layer_request(
        extra.get("output_residual_stream") if extra else None
    )
    qk, qk_layers = _parse_layer_request(extra.get("output_qk") if extra else None)
    return _ReqPlan(
        gen=gen,
        sp_id=id(sp),
        steer=configs,
        steer_layers=frozenset(layers),
        steer_broadcast=broadcast,
        steer_min_pos=min_pos,
        steer_max_pos=max_pos,
        hooks=hooks,
        pre_layers=pre_layers,
        post_layers=post_layers,
        cap=cap,
        cap_layers=cap_layers,
        qk=qk,
        qk_layers=qk_layers,
    )


@dataclass(slots=True)
class _StepPlan:
    """Everything the hooks need for one forward pass."""

    step: int
    num_reqs: int
    req_ids: list[str]
    qsl: list[int]
    """``query_start_loc[: num_reqs + 1]`` as host ints."""
    abs_start: list[int]
    """Absolute position of each row's first scheduled token."""
    rows: list[_ReqPlan | None]
    steer: dict[int, list[int]] = field(default_factory=dict)
    """layer -> rows whose steering can touch this pass."""
    pre: dict[int, list[int]] = field(default_factory=dict)
    post: dict[int, list[int]] = field(default_factory=dict)
    cap_all: list[int] = field(default_factory=list)
    cap_by_layer: dict[int, list[int]] = field(default_factory=dict)
    qk: list[int] = field(default_factory=list)
    active: bool = False
    """Some hook has work this pass (it must not replay a CUDA graph)."""
    cap_cache: dict[tuple[int, ...], Any] = field(default_factory=dict)
    """Capture row set -> (token selection, segments), shared by the layers."""

    def capture_rows(self, layer_idx: int) -> list[int]:
        rows = self.cap_by_layer.get(layer_idx)
        if not self.cap_all:
            return rows or []
        if not rows:
            return self.cap_all
        return sorted(set(self.cap_all) | set(rows))


def _host_query_start_loc(runner: Any, num_reqs: int) -> list[int]:
    """``query_start_loc[: num_reqs + 1]`` from the runner's pinned host mirror.

    Falls back to the attention metadata (one device sync per pass) on
    runners without the mirror.
    """
    host = getattr(getattr(runner, "query_start_loc", None), "np", None)
    if host is not None:
        return host[: num_reqs + 1].tolist()
    if not is_forward_context_available():
        raise RuntimeError("vllm-lens: no host query_start_loc and no forward context")
    meta = get_forward_context().attn_metadata
    if isinstance(meta, list):
        meta = meta[0] if meta else None
    for m in (meta or {}).values():
        q = getattr(m, "query_start_loc", None)
        if q is not None:
            return [int(x) for x in q[: num_reqs + 1].tolist()]
    raise RuntimeError("vllm-lens: no host query_start_loc and no attention metadata")


def _build_step_plan(
    extension: HiddenStatesExtension, runner: Any, step: int
) -> _StepPlan | None:
    batch = runner.input_batch
    num_reqs = int(batch.num_reqs)
    if num_reqs == 0:
        return None
    req_ids = list(batch.req_ids[:num_reqs])
    qsl = _host_query_start_loc(runner, num_reqs)
    # Positions of a pass are num_computed_tokens + 0..n-1 (the runner builds
    # them from the same buffer), so this is each row's absolute start.
    abs_start = batch.num_computed_tokens_cpu[:num_reqs].tolist()
    plan = _StepPlan(step, num_reqs, req_ids, qsl, abs_start, [])
    cache = extension._req_plans
    gen = extension._lens_gen
    rows = plan.rows
    requests = runner.requests
    for req_id in req_ids:
        rp = cache.get(req_id)
        state = requests.get(req_id)
        if rp is None or rp.gen != gen or (
            state is not None and rp.sp_id != id(state.sampling_params)
        ):
            try:
                rp = _resolve_request(extension, runner, req_id, gen)
            except Exception:
                # One malformed request must not take the batch (or the
                # engine, when planning runs inside batch dispatch) down.
                _log_hook_error(extension, "request resolution", -1)
                rp = None
            if rp is not None:
                cache[req_id] = rp
        rows.append(rp)
    if getattr(runner, "use_async_spec_decode", False) and any(
        rp is not None and rp.steer and not rp.steer_broadcast for rp in rows
    ):
        # With async speculative decoding the host counts assume every draft
        # token was accepted; the runner corrects them on the device.
        abs_start = runner.num_computed_tokens[:num_reqs].tolist()
        plan.abs_start = abs_start
    for i, rp in enumerate(rows):
        if rp is None or rp.idle:
            continue
        if rp.steer:
            a0 = abs_start[i]
            n = qsl[i + 1] - qsl[i]
            if rp.steer_broadcast or (
                rp.steer_max_pos >= a0 and rp.steer_min_pos < a0 + n
            ):
                for layer_idx in rp.steer_layers:
                    plan.steer.setdefault(layer_idx, []).append(i)
        for layer_idx in rp.pre_layers:
            plan.pre.setdefault(layer_idx, []).append(i)
        for layer_idx in rp.post_layers:
            plan.post.setdefault(layer_idx, []).append(i)
        if rp.cap:
            if rp.cap_layers is None:
                plan.cap_all.append(i)
            else:
                for layer_idx in rp.cap_layers:
                    plan.cap_by_layer.setdefault(layer_idx, []).append(i)
        if rp.qk:
            plan.qk.append(i)
    plan.active = bool(
        plan.steer
        or plan.pre
        or plan.post
        or plan.cap_all
        or plan.cap_by_layer
        or plan.qk
        or extension._persistent_hooks
    )
    if len(cache) > num_reqs + _REQ_CACHE_SLACK:
        live = runner.requests
        for stale in [k for k in cache if k not in live]:
            del cache[stale]
    stats = extension._lens_stats
    stats["passes_planned"] += 1
    if plan.active:
        stats["passes_active"] += 1
    return plan


def _current_plan(extension: HiddenStatesExtension) -> _StepPlan | None:
    """The plan for the forward pass in progress (built by its first caller).

    Passes outside ``execute_model`` (warm-up, profiling, CUDA-graph capture)
    get ``None``, so the hooks never act on dummy batches or bake anything
    into a captured graph.
    """
    if not extension._in_execute:
        return None
    plan = extension._step_plan
    if plan is not None and plan.step == extension._exec_step:
        return plan
    # No forward-context requirement: the dispatch wrapper plans the pass
    # before vLLM sets the context, and planning reads only host-side state.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    plan = _build_step_plan(extension, extension.model_runner, extension._exec_step)
    extension._step_plan = plan
    return plan


def _install_runner_wrappers(extension: HiddenStatesExtension) -> None:
    """Wrap the model runner so each forward pass is planned once.

    ``execute_model`` is bracketed so the hooks know which pass they are in
    (and ignore warm-up / capture passes).  ``_determine_batch_execution_and_padding``
    (which picks CUDA-graph replay vs eager for a pass) is asked to run eagerly
    exactly when the pass's plan has hook work, so passes without any keep
    replaying CUDA graphs while hooked passes still run every hook.
    """
    runner = extension.model_runner
    if getattr(runner, "_vllm_lens_wrapped", False):
        return
    orig_execute = runner.execute_model

    def execute_model(*args: Any, **kwargs: Any) -> Any:
        if extension._capture_blocks:
            # File copies that have landed, so pinned staging memory is
            # reused pass after pass instead of holding every capture.
            extension._flush_capture_blocks(wait=False)
        extension._exec_step += 1
        extension._step_plan = None
        extension._in_execute = True
        try:
            return orig_execute(*args, **kwargs)
        finally:
            extension._in_execute = False
            extension._step_plan = None

    runner.execute_model = execute_model
    orig_determine = getattr(runner, "_determine_batch_execution_and_padding", None)
    if orig_determine is not None:

        def determine(*args: Any, **kwargs: Any) -> Any:
            if extension._in_execute and not kwargs.get("force_eager", False):
                try:
                    plan = _current_plan(extension)
                    eager = plan is not None and plan.active
                except Exception:
                    # Planning failed: run eagerly so the hooks can still try.
                    _log_hook_error(extension, "pass planning", -1)
                    extension._step_plan = None
                    eager = True
                if eager:
                    kwargs["force_eager"] = True
                    extension._lens_stats["passes_forced_eager"] += 1
            return orig_determine(*args, **kwargs)

        runner._determine_batch_execution_and_padding = determine
    extension._dispatch_wrapped = orig_determine is not None
    runner._vllm_lens_wrapped = True


def _cuda_graphs_enabled(extension: HiddenStatesExtension) -> bool:
    """True when some passes may replay CUDA graphs (where hooks cannot run)."""
    try:
        cfg = extension.model_runner.vllm_config
        if cfg.model_config.enforce_eager:
            return False
        mode = cfg.compilation_config.cudagraph_mode
        return mode is not None and getattr(mode, "name", str(mode)) != "NONE"
    except Exception:  # pragma: no cover - defensive against config drift
        return True


def _log_hook_error(extension: HiddenStatesExtension, kind: str, layer_idx: int) -> None:
    """Log a swallowed hook error: the first few with tracebacks, then sparsely.

    A failing hook fires on every layer of every pass, so logging each one
    would flood the log and slow the engine down.
    """
    stats = extension._lens_stats
    stats["hook_errors"] += 1
    exc = sys.exc_info()[1]
    key = (kind, type(exc).__name__ if exc is not None else "")
    counts = extension._error_counts
    n = counts[key] = counts.get(key, 0) + 1
    if n <= _ERROR_LOG_FIRST or n % _ERROR_LOG_EVERY == 0:
        logger.warning(
            "vllm-lens %s error on layer %d, skipping (occurrence %d of this "
            "error; after the first %d only every %d-th is logged)",
            kind,
            layer_idx,
            n,
            _ERROR_LOG_FIRST,
            _ERROR_LOG_EVERY,
            exc_info=True,
        )


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    norm_ref: torch.Tensor | None,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``norm_ref`` is the tensor whose per-token L2 norm ``norm_match`` should
    match.  For fused-residual models (e.g. Qwen3) the layer returns
    ``(hidden_states, residual)`` and ``target`` is only ``hidden_states``
    (the MLP-delta half); the *true* residual stream is
    ``hidden_states + residual``, so ``norm_ref`` must be that full stream,
    not the MLP-delta half, or the steering vector is scaled to a far smaller
    norm than HF uses.  For non-fused / plain-tensor layers the caller passes
    ``norm_ref = target``.  Required (no default) so a forgotten reference
    fails at the call instead of silently scaling to the wrong
    MLP-delta-half norm; ``None`` only when no config uses ``norm_match``.
    """
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)  # (hidden,) or (n_pos, hidden)

        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                assert norm_ref is not None
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else list(range(vec.shape[0]))
            )
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    assert norm_ref is not None
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a small host tensor to ``device`` without blocking the host.

    ``torch.tensor(..., device="cuda")`` copies from pageable memory and
    waits for the stream to drain, which stalls the CPU on every layer that
    builds an index.  A pinned staging copy keeps the transfer asynchronous.
    """
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def _norm_ref_rows(
    target: torch.Tensor,
    base: tuple[torch.Tensor, torch.Tensor | None] | None,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Rows ``idx`` of the tensor ``norm_match`` measures (see ``_apply_steering``)."""
    if base is None:
        return target.index_select(0, idx)
    hidden, residual = base
    ref = hidden.index_select(0, idx)
    return ref if residual is None else ref + residual.index_select(0, idx)


def _norm_ref_full(
    target: torch.Tensor, base: tuple[torch.Tensor, torch.Tensor | None] | None
) -> torch.Tensor:
    if base is None:
        return target
    hidden, residual = base
    return hidden if residual is None else hidden + residual


def _steer_layer(
    plan: _StepPlan,
    rows: list[int],
    layer_idx: int,
    target: torch.Tensor,
    base: tuple[torch.Tensor, torch.Tensor | None] | None,
) -> None:
    """Apply this pass's steering at one layer.

    ``target`` is the cloned stream half being edited.  ``base`` holds the
    layer's original ``(hidden_states, residual)`` on fused-residual layers
    (``norm_match`` measures their sum, the full residual stream), or
    ``None`` for plain-tensor layers (it measures ``target``).

    Rows that receive at most one vector per token (decode steps and
    position-specific vectors: the common case) are applied together with one
    ``index_add_``: the same multiply and add per element as
    ``_apply_steering``, so without ``norm_match`` the result is bit-identical.
    With ``norm_match`` the token norms come from one batched reduction,
    which can differ from a per-token reduction in the last fp32 bit.  Rows
    where several vectors land on one token, or where a broadcast vector
    covers a multi-token chunk, go through ``_apply_steering`` row by row.
    """
    qsl, abs_start = plan.qsl, plan.abs_start
    flat_rows: list[int] = []
    vecs: list[torch.Tensor] = []
    scales: list[float] = []
    norms: list[bool] = []
    sequential: list[int] = []
    for i in rows:
        rp = plan.rows[i]
        assert rp is not None
        start, end, a0 = qsl[i], qsl[i + 1], abs_start[i]
        n_tokens = end - start
        r_rows: list[int] = []
        r_vecs: list[torch.Tensor] = []
        r_cfgs: list[SteeringVector] = []
        simple = True
        for cfg in rp.steer:
            act_idx = cfg.layer_index_map.get(layer_idx)
            if act_idx is None:
                continue
            vec = cfg.activations[act_idx]
            if vec.dim() == 1:
                if n_tokens != 1:
                    simple = False
                    break
                r_rows.append(start)
                r_vecs.append(vec)
                r_cfgs.append(cfg)
                continue
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else range(vec.shape[0])
            )
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if a0 <= abs_pos < a0 + n_tokens:
                    r_rows.append(abs_pos - a0 + start)
                    r_vecs.append(vec[pi])
                    r_cfgs.append(cfg)
        if not simple or len(set(r_rows)) != len(r_rows):
            sequential.append(i)
            continue
        flat_rows.extend(r_rows)
        vecs.extend(r_vecs)
        scales.extend(c.scale for c in r_cfgs)
        norms.extend(c.norm_match for c in r_cfgs)

    if flat_rows:
        device = target.device
        idx = _to_device(torch.tensor(flat_rows, dtype=torch.long), device)
        v = torch.stack([x.to(target.dtype) for x in vecs])
        if any(norms):
            ref = _norm_ref_rows(target, base, idx)
            r_norm = ref.float().norm(dim=-1, keepdim=True)
            v_norm = v.float().norm(dim=-1, keepdim=True)
            matched = (v * (r_norm / (v_norm + 1e-6))).to(ref.dtype)
            if all(norms):
                v = matched
            else:
                mask = _to_device(torch.tensor(norms), device).unsqueeze(1)
                v = torch.where(mask, matched, v)
        if all(s == scales[0] for s in scales):
            v = v * scales[0]
        else:
            s = _to_device(torch.tensor(scales, dtype=torch.float32), device)
            v = (v.float() * s.unsqueeze(1)).to(target.dtype)
        target.index_add_(0, idx, v)
    if sequential:
        needs_norm = any(
            cfg.norm_match
            for i in sequential
            for cfg in plan.rows[i].steer  # type: ignore[union-attr]
        )
        ref = _norm_ref_full(target, base) if needs_norm else None
        for i in sequential:
            _apply_steering(
                plan.rows[i].steer,  # type: ignore[union-attr]
                layer_idx,
                target,
                qsl[i],
                qsl[i + 1],
                abs_start[i],
                ref,
            )


@dataclass(slots=True)
class _CaptureBlock:
    """One layer's captured rows of one pass, copied to pinned host memory.

    ``segments`` lists ``(internal_req_id, n_rows)`` back to back in
    ``host``; ``event`` completes when the asynchronous copy has landed.
    """

    layer_idx: int
    host: torch.Tensor
    event: Any
    segments: list[tuple[str, int]]


def _capture_selection(
    plan: _StepPlan, rows: list[int], device: torch.device
) -> tuple[tuple[int, int] | torch.Tensor, list[tuple[str, int]]]:
    """Token selection for a set of capturing rows (cached for the pass).

    Returns ``(span, segments)`` where ``span`` is ``(start, end)`` when the
    rows' tokens are contiguous, else a device index tensor.
    """
    key = tuple(rows)
    hit = plan.cap_cache.get(key)
    if hit is not None:
        return hit
    qsl = plan.qsl
    spans = [(qsl[i], qsl[i + 1]) for i in rows]
    segments = [(plan.req_ids[i], e - s) for i, (s, e) in zip(rows, spans)]
    if all(spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1)):
        sel: tuple[int, int] | torch.Tensor = (spans[0][0], spans[-1][1])
    else:
        idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in spans])
        sel = _to_device(torch.from_numpy(idx), device)
    plan.cap_cache[key] = (sel, segments)
    return sel, segments


def _capture_layer(
    extension: HiddenStatesExtension,
    plan: _StepPlan,
    layer_idx: int,
    stream: torch.Tensor,
    residual: torch.Tensor | None,
    rows: list[int],
) -> None:
    """Copy the capturing rows of one layer to the host in one transfer.

    Gathers every capturing request's tokens (one slice when they are
    contiguous, else one ``index_select``), forms ``stream + residual`` on
    those rows only, and enqueues one asynchronous copy into pinned memory.
    The rows are split per request when they are retrieved.
    """
    span, segments = _capture_selection(plan, rows, stream.device)
    if isinstance(span, tuple):
        s0, e0 = span
        sel = stream[s0:e0]
        if residual is not None:
            sel = sel + residual[s0:e0]
    else:
        sel = stream.index_select(0, span)
        if residual is not None:
            sel += residual.index_select(0, span)
    if sel.device.type == "cuda":
        host = torch.empty(sel.shape, dtype=sel.dtype, pin_memory=True)
        host.copy_(sel, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
    else:
        host, event = sel.to("cpu", copy=True), None
    extension._capture_blocks.append(_CaptureBlock(layer_idx, host, event, segments))
    extension._lens_stats["capture_copies"] += 1


def _split_output(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    plan = _current_plan(extension)
    if plan is None or not plan.active:
        return None
    steer_rows = plan.steer.get(layer_idx)
    persistent_post = layer_idx in extension._persistent_post_layers
    post_rows = range(plan.num_reqs) if persistent_post else plan.post.get(layer_idx)
    cap_rows = plan.capture_rows(layer_idx) if extension._should_capture else []
    if not steer_rows and not post_rows and not cap_rows:
        return None
    qsl = plan.qsl

    # --- Phase 1: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if steer_rows:
        base: tuple[torch.Tensor, torch.Tensor | None] | None
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), *output[1:])
            target = modified_output[0]
            # Fused-residual layers: true residual stream is output[0]+output[1].
            # norm_match must reference the full stream, not the MLP-delta half.
            base = (output[0], output[1])
        else:
            modified_output = output.clone()
            target = modified_output
            base = None
        _steer_layer(plan, steer_rows, layer_idx, target, base)
        extension._lens_stats["steered_layer_passes"] += 1

    # --- Phase 2: run generic (post) hooks ----------------------------
    # Per-request and persistent hooks are stored in separate context
    # dicts (per-request contexts are cleaned up after each request;
    # persistent ones accumulate).  Within each dict, contexts are keyed
    # by the hook's position in its category list — so a pre-hook and a
    # post-hook at different positions never share a HookContext, and the
    # returned result index ("0", "1", ...) is stable regardless of how
    # many pre/post hooks a request mixes.  Pre-hooks are handled in
    # _pre_hook_inner using the same position keys.
    if post_rows:
        runner = extension.model_runner
        # Compute hidden_states (summed if tuple) same as Phase 3 does.
        hook_src = modified_output if modified_output is not None else output
        if isinstance(hook_src, tuple):
            hook_hidden = (
                hook_src[0] + hook_src[1] if hook_src[1] is not None else hook_src[0]
            )
        else:
            hook_hidden = hook_src
        # Clone to avoid aliasing — hooks read/write this independently.
        hook_hidden = hook_hidden.clone()

        def _run_post_category(
            hooks: list[Hook],
            store: dict[str, dict[int, HookContext]],
            req_id: str,
            start: int,
            end: int,
            abs_start: int = 0,
        ) -> None:
            """Run the post-hooks in one category list at this layer.

            Contexts live in ``store[req_id][position]``, created lazily.
            """
            nonlocal modified_output
            for pos, hook in enumerate(hooks):
                if hook.pre or not hook.has_layer(layer_idx):
                    continue
                ctxs = store.setdefault(req_id, {})
                ctx = ctxs.get(pos)
                if ctx is None:
                    ctx = HookContext()
                    ctxs[pos] = ctx
                ctx.layer_idx = layer_idx
                ctx.seq_len = end - start
                ctx.abs_start = abs_start
                ctx.model = _runner_model(runner)
                ctx._prefetched = extension._prefetched_params

                result = hook.fn(ctx, hook_hidden[start:end])
                if result is not None:
                    modified_output = _replace_hook_output(
                        output, modified_output, hook_hidden, start, end, result
                    )

        persistent_hooks = extension._persistent_hooks
        for i in post_rows:
            rp = plan.rows[i]
            if rp is None:
                continue
            req_id = plan.req_ids[i]
            start, end = qsl[i], qsl[i + 1]
            # Persistent hooks fire first (base layer); per-request hooks
            # see the persistent-modified state.
            a0 = plan.abs_start[i]  # 018 patch: absolute start of this request's rows
            _run_post_category(
                persistent_hooks,
                extension._persistent_hook_contexts,
                req_id,
                start,
                end,
                a0,
            )
            _run_post_category(rp.hooks, extension._hook_contexts, req_id, start, end, a0)

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if cap_rows:
        stream, residual = _split_output(
            modified_output if modified_output is not None else output
        )
        _capture_layer(extension, plan, layer_idx, stream, residual, cap_rows)

    return modified_output


@dataclass(slots=True, frozen=True)
class _LayerInputs:
    """Where a decoder layer's stream inputs sit in its call, from its signature.

    ``hidden`` / ``residual`` are ``(parameter name, positional index)``.
    ``fused`` marks vLLM's fused-residual layers (``if residual is None:``
    in ``forward``), whose stream entering the layer is
    ``hidden_states + residual``; other layers (Command-R, PhiMoE, EXAONE-4
    pass a stale residual they ignore) are treated as single-tensor inputs.
    """

    hidden: tuple[str, int]
    residual: tuple[str, int] | None
    fused: bool


def _layer_inputs(layer: torch.nn.Module) -> _LayerInputs:
    forward = type(layer).forward
    try:
        params = [
            p
            for p in inspect.signature(forward).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ][1:]  # drop self
        names = [p.name for p in params]
    except (TypeError, ValueError):
        names = []
    if "hidden_states" in names:
        hidden = ("hidden_states", names.index("hidden_states"))
    elif "x" in names:
        hidden = ("x", names.index("x"))
    else:  # vLLM's (positions, hidden_states, residual) convention
        hidden = (names[1] if len(names) > 1 else "hidden_states", 1)
    residual = ("residual", names.index("residual")) if "residual" in names else None
    try:
        fused = residual is not None and "residual is None" in inspect.getsource(forward)
    except (OSError, TypeError):
        fused = False
    return _LayerInputs(hidden, residual, fused)


def _call_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], slot: tuple[str, int]
) -> Any:
    name, index = slot
    if name in kwargs:
        return kwargs[name]
    return args[index] if index < len(args) else None


def _with_call_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], slot: tuple[str, int], value: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    name, index = slot
    if name in kwargs or index >= len(args):
        kwargs = dict(kwargs)
        kwargs[name] = value
        return args, kwargs
    new_args = list(args)
    new_args[index] = value
    return tuple(new_args), kwargs


def _pre_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    spec: _LayerInputs,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Run pre-hooks (hook.pre=True) on the residual stream entering a layer.

    The hook sees the stream entering the layer: ``hidden_states`` at layer 0
    (and on layers that are not fused-residual), and ``hidden_states +
    residual`` on fused-residual layers after it (the same tensor the
    previous layer's post-hooks see).  A returned tensor replaces the
    request's rows of that stream; edit by returning, not in place.  Only
    runs generic hooks — steering and activation capture are post-hook
    operations and are not affected.
    """
    plan = _current_plan(extension)
    if plan is None or not plan.active:
        return None
    persistent_pre = layer_idx in extension._persistent_pre_layers
    rows = range(plan.num_reqs) if persistent_pre else plan.pre.get(layer_idx)
    if not rows:
        return None

    hidden = _call_arg(args, kwargs, spec.hidden)
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError(
            f"vllm-lens: layer {layer_idx} input {spec.hidden[0]!r} is "
            f"{type(hidden).__name__}, not a tensor"
        )
    residual = _call_arg(args, kwargs, spec.residual) if spec.fused else None
    if not isinstance(residual, torch.Tensor):
        residual = None
    runner = extension.model_runner
    # Pre-hooks share the same context stores as post-hooks, keyed by the
    # hook's position in its category list.  A hook at a given position is
    # either pre or post (never both), so pre and post never collide on the
    # same key — this is what lets a request mix pre- and post-hooks safely.
    stream = hidden if residual is None else hidden + residual
    new_hidden: torch.Tensor | None = None
    new_residual: torch.Tensor | None = None

    def _run_pre_category(
        hooks: list[Hook],
        store: dict[str, dict[int, HookContext]],
        req_id: str,
        start: int,
        end: int,
    ) -> None:
        """Run the pre-hooks in one category list at this layer."""
        nonlocal stream, new_hidden, new_residual
        for pos, hook in enumerate(hooks):
            if not hook.pre or not hook.has_layer(layer_idx):
                continue
            ctxs = store.setdefault(req_id, {})
            hctx = ctxs.get(pos)
            if hctx is None:
                hctx = HookContext()
                ctxs[pos] = hctx
            hctx.layer_idx = layer_idx
            hctx.seq_len = end - start
            hctx.model = _runner_model(runner)
            hctx._prefetched = extension._prefetched_params

            result = hook.fn(hctx, stream[start:end])
            if result is None:
                continue
            if new_hidden is None:
                new_hidden = hidden.clone()
                if residual is None:
                    stream = new_hidden  # later hooks see the replaced rows
                else:
                    new_residual = residual.clone()
                    # Fresh copy: views earlier hooks kept stay unchanged.
                    stream = stream.clone()
            new_hidden[start:end] = result
            if residual is not None:
                # Store the requested stream in hidden_states and zero the
                # residual rows: the layer's fused add then reproduces it.
                assert new_residual is not None
                new_residual[start:end] = 0
                stream[start:end] = result

    persistent_hooks = extension._persistent_hooks
    for i in rows:
        rp = plan.rows[i]
        if rp is None:
            continue
        req_id = plan.req_ids[i]
        start, end = plan.qsl[i], plan.qsl[i + 1]
        _run_pre_category(
            persistent_hooks, extension._persistent_hook_contexts, req_id, start, end
        )
        _run_pre_category(rp.hooks, extension._hook_contexts, req_id, start, end)

    if new_hidden is None:
        return None
    args, kwargs = _with_call_arg(args, kwargs, spec.hidden, new_hidden)
    if new_residual is not None:
        assert spec.residual is not None
        args, kwargs = _with_call_arg(args, kwargs, spec.residual, new_residual)
    return args, kwargs


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            _log_hook_error(extension, "hook", layer_idx)
            return None

    return hook


def _make_pre_hook(
    extension: HiddenStatesExtension, layer_idx: int, spec: _LayerInputs
) -> Callable:
    """Create a forward pre-hook closure for a specific layer index.

    Registered with ``with_kwargs=True``: model code passes the layer inputs
    positionally (Llama, Qwen2) or by keyword (Qwen3-Next, Qwen3.5 / 3.6), so
    they are located by the layer's signature (``_layer_inputs``).  The first
    layer's pre-hook also plans the pass (``_current_plan``).
    """

    def hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Forward pre-hook: run user pre-hooks on the layer input."""
        try:
            return _pre_hook_inner(extension, layer_idx, spec, args, kwargs)
        except Exception:
            _log_hook_error(extension, "pre-hook", layer_idx)
            return None

    return hook


def _warn_qk_skipped(extension: HiddenStatesExtension, reason: str) -> None:
    """Log once per worker when Q/K capture has to bail out of a forward.

    The hook otherwise returns silently and the client only sees a missing
    ``qk_layers`` key, which points at the wrong cause.
    """
    if extension._qk_skip_warned:
        return
    extension._qk_skip_warned = True
    logger.warning(
        "vllm-lens: output_qk capture skipped a forward pass (%s); attention "
        "Q/K for this request may be incomplete or missing.",
        reason,
    )


def _qk_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    attn_module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Capture post-RoPE Q/K at the input of a vLLM ``Attention`` module.

    Unlike residual-stream capture, this runs on **every** TP rank:
    ``Attention`` inputs are sharded by head (each rank sees only its own
    slice of query/kv heads), so all ranks must contribute and the plugin
    concatenates along the head dimension at merge time.
    """
    plan = _current_plan(extension)
    if plan is None or not plan.qk:
        return

    query = kwargs.get("query", args[0] if len(args) > 0 else None)
    key = kwargs.get("key", args[1] if len(args) > 1 else None)
    if query is None or key is None:
        _warn_qk_skipped(extension, "attention inputs carry no query/key")
        return

    # Inputs are (num_tokens, heads * head_size), post-RoPE (the model
    # applies rotary embeddings before calling the Attention wrapper);
    # 3-D inputs are legal per the wrapper's contract, so reshape both.
    num_heads = attn_module.num_heads
    num_kv_heads = attn_module.num_kv_heads
    head_size = attn_module.head_size
    q3 = query.reshape(-1, num_heads, head_size)
    k3 = key.reshape(-1, num_kv_heads, head_size)

    for i in plan.qk:
        rp = plan.rows[i]
        if rp is None or (rp.qk_layers is not None and layer_idx not in rp.qk_layers):
            continue
        req_id = plan.req_ids[i]
        start, end = plan.qsl[i], plan.qsl[i + 1]

        if req_id not in extension._captured_qk:
            extension._captured_qk[req_id] = {}
        layer_states = extension._captured_qk[req_id]
        if layer_idx not in layer_states:
            layer_states[layer_idx] = {"q": [], "k": []}
        layer_states[layer_idx]["q"].append(q3[start:end].cpu())
        layer_states[layer_idx]["k"].append(k3[start:end].cpu())


def _make_qk_hook(
    extension: HiddenStatesExtension, layer_idx: int, attn_module: Any
) -> Callable:
    """Create a forward pre-hook closure for an ``Attention`` module."""

    def hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Forward pre-hook: capture this layer's Q/K for opted-in requests."""
        try:
            _qk_hook_inner(extension, layer_idx, attn_module, args, kwargs)
        except Exception:
            _log_hook_error(extension, "qk hook", layer_idx)
        return None

    return hook


def _normalize_sliding_window(value: Any) -> tuple[int, int]:
    """Normalize a sliding-window spec to vLLM's ``(left, right)`` form."""
    if value is None:
        return (-1, -1)
    if isinstance(value, int):
        return (-1, -1) if value < 0 else (value - 1, 0)
    left, right = value
    return (int(left), int(right))


def _to_float_list(value: Any) -> list[float] | None:
    """Convert an optional per-head tensor/sequence to a JSON-safe list."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().float().cpu().tolist()]
    return [float(x) for x in value]


def _new_stats() -> dict[str, int]:
    return {
        "passes_planned": 0,
        "passes_active": 0,
        "passes_forced_eager": 0,
        "steered_layer_passes": 0,
        "capture_copies": 0,
        "hook_errors": 0,
        "retrieval_flush_s": 0.0,
        "retrieval_build_s": 0.0,
        "retrieval_pickle_s": 0.0,
    }


def _by_external_id(
    internal_ids: list[str], external_ids: list[str]
) -> dict[str, list[str]]:
    """Group internal request ids under the external id they belong to.

    Same matching as upstream's per-id scan (``startswith(f"{ext}-")``, the
    first external id in ``external_ids`` order wins), in linear time.
    """
    rank = {}
    for n, ext in enumerate(external_ids):
        rank.setdefault(ext, n)
    out: dict[str, list[str]] = {}
    for internal in internal_ids:
        best = None
        for k in _prefix_keys(internal):
            r = rank.get(k)
            if r is not None and (best is None or r < rank[best]):
                best = k
        if best is not None:
            out.setdefault(best, []).append(internal)
    return out


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        model_config: Any
        compilation_config: Any
        rank: int
        parallel_config: ParallelConfig

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

    # Per-request hook definitions:
    # key (external_req_id or _hook_id) → list of Hook
    _hook_data: dict[str, list[Hook]] = {}

    # Persistent hooks (apply to every request, not auto-cleaned):
    _persistent_hooks: list[Hook] = []

    # Per-request hook contexts, keyed by internal request ID then by the
    # hook's position in the per-request hook list:
    # internal_req_id → { hook_position → HookContext }
    _hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Persistent hook contexts (separate from per-request to avoid cleanup
    # conflicts), keyed the same way by position in the persistent hook list:
    # internal_req_id → { hook_position → HookContext }
    _persistent_hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    # Per-request captured attention Q/K (all TP ranks — heads are sharded):
    # internal_req_id → { layer_idx → {"q": [tensor, ...], "k": [tensor, ...]} }
    _captured_qk: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {}
    _qk_hooks_installed: bool = False
    _qk_skip_warned: bool = False

    # Per-layer kernel metadata recorded at install time, keyed by global
    # layer index (scale, sliding window, soft-cap, ... — see install_qk_hooks).
    _qk_layer_meta: dict[int, dict[str, Any]] = {}

    # Pass bookkeeping (see _install_runner_wrappers / _current_plan).
    _in_execute: bool = False
    _exec_step: int = 0
    _step_plan: _StepPlan | None = None
    _dispatch_wrapped: bool = False

    # Bumped whenever registered steering / hook data changes; invalidates
    # the cached per-request resolutions in _req_plans.
    _lens_gen: int = 0
    _req_plans: dict[str, _ReqPlan] = {}
    # Insertion order of the keys of _steering_data / _hook_data, which
    # decides the order in which several matching entries are applied.
    _steering_order: dict[str, int] = {}
    _hook_order: dict[str, int] = {}
    # Device->host copies of captured rows not yet filed per request.
    _capture_blocks: list[_CaptureBlock] = []
    # Layers that persistent pre-/post-hooks touch (they apply to every row).
    _persistent_pre_layers: frozenset[int] = frozenset()
    _persistent_post_layers: frozenset[int] = frozenset()
    _lens_stats: dict[str, int] = _new_stats()
    _error_counts: dict[tuple[str, str], int] = {}

    def _init_lens_state(self) -> None:
        """Give this worker its own state containers (idempotent).

        The class attributes above are shared defaults; every container is
        replaced by an instance attribute on first use so state set before
        ``install_hooks`` is kept.
        """
        if self.__dict__.get("_lens_state_ready"):
            return
        d = self.__dict__
        for name, factory in (
            ("_captured_states", dict),
            ("_steering_data", dict),
            ("_steering_order", dict),
            ("_hook_data", dict),
            ("_hook_order", dict),
            ("_hook_contexts", dict),
            ("_persistent_hook_contexts", dict),
            ("_captured_qk", dict),
            ("_qk_layer_meta", dict),
            ("_req_plans", dict),
            ("_capture_blocks", list),
            ("_prefetched_params", dict),
            ("_error_counts", dict),
        ):
            if name not in d:
                setattr(self, name, factory())
        if not isinstance(d.get("_persistent_hooks"), list):
            self._persistent_hooks = []
        self._refresh_persistent_layers()
        self._lens_stats = _new_stats()
        self._order_seq = 0
        self._lens_state_ready = True

    def _bump_gen(self) -> None:
        """Registered data changed: every cached resolution is stale.

        Dropping them (rather than waiting for the size-based pruning) also
        releases the device tensors of cleared steering vectors at once.
        """
        self._lens_gen += 1
        self._req_plans.clear()

    def _refresh_persistent_layers(self) -> None:
        hooks = self._persistent_hooks
        self._persistent_pre_layers = frozenset(
            i for h in hooks if h.pre for i in h.layer_indices
        )
        self._persistent_post_layers = frozenset(
            i for h in hooks if not h.pre for i in h.layer_indices
        )

    def _install_pass_tracking(self) -> None:
        _install_runner_wrappers(self)
        if _cuda_graphs_enabled(self) and not self._dispatch_wrapped:
            raise RuntimeError(
                "vllm-lens: CUDA graphs are enabled but this vLLM model runner has no "
                "_determine_batch_execution_and_padding to route hooked passes to "
                "eager execution, so hooks would silently skip decode steps. Set "
                "VLLM_LENS_CUDA_GRAPHS=0 (eager mode) or pin vLLM 0.30.0."
            )

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Hooks only fire in passes that run the layers eagerly: the plugin
        turns torch.compile off, and passes with hook work are routed to
        eager execution (``_install_runner_wrappers``) while the rest may
        replay CUDA graphs.
        """
        if self._hooks_installed:
            return
        self._init_lens_state()
        runner = self.model_runner
        if not hasattr(runner, "input_batch") or not hasattr(runner, "requests"):
            raise RuntimeError(
                f"vllm-lens: model runner {type(runner).__name__} has no V1 per-step "
                "state (input_batch / requests); this is vLLM's V2 model runner. "
                "Run with VLLM_USE_V2_MODEL_RUNNER=0 (the plugin's default)."
            )

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layer_map = _discover_layer_modules(self)
        self._install_pass_tracking()
        for layer_idx, layer in sorted(layer_map.items()):
            layer.register_forward_pre_hook(
                _make_pre_hook(self, layer_idx, _layer_inputs(layer)), with_kwargs=True
            )
            layer.register_forward_hook(_make_hook(self, layer_idx))
        self._hooks_installed = True
        logger.info(
            "vllm-lens: hooks installed on %d decoder layer(s)%s",
            len(layer_map),
            "; CUDA graphs stay on for passes without hook work"
            if _cuda_graphs_enabled(self)
            else "",
        )

    def lens_stats(self, reset: bool = False) -> dict[str, Any]:
        """Counters for diagnostics: planned, hooked and forced-eager passes, errors."""
        self._init_lens_state()
        out: dict[str, Any] = dict(self._lens_stats)
        out["cuda_graphs"] = _cuda_graphs_enabled(self)
        out["hooks_installed"] = bool(self._hooks_installed)
        out["qk_hooks_installed"] = bool(self._qk_hooks_installed)
        if reset:
            self._lens_stats = _new_stats()
        return out

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def _prepare_steering(
        self, payload: dict[str, list[SteeringVector]]
    ) -> dict[str, list[SteeringVector]]:
        """Validate layer indices and move every activation tensor to the device.

        Tensors with the same shape and dtype are moved in one stacked copy.
        """
        param = next(_runner_model(self.model_runner).parameters())
        device, dtype = param.device, param.dtype
        num_layers = _get_total_num_layers(self)
        groups: dict[tuple, list[tuple[str, int, torch.Tensor]]] = {}
        for key, sv_list in payload.items():
            for j, sv in enumerate(sv_list):
                for idx in sv.layer_indices:
                    if idx < 0 or idx >= num_layers:
                        raise ValueError(
                            f"layer_index {idx} out of range [0, {num_layers})"
                        )
                t = sv.activations
                groups.setdefault((tuple(t.shape), t.dtype, t.device), []).append(
                    (key, j, t)
                )
        moved: dict[tuple[str, int], torch.Tensor] = {}
        for members in groups.values():
            if len(members) == 1:
                key, j, t = members[0]
                moved[(key, j)] = t.to(device=device, dtype=dtype)
                continue
            block = torch.stack([t for _, _, t in members]).to(device=device, dtype=dtype)
            for (key, j, _), t in zip(members, block.unbind(0)):
                moved[(key, j)] = t
        return {
            key: [
                sv.model_copy(update={"activations": moved[(key, j)]})
                for j, sv in enumerate(sv_list)
            ]
            for key, sv_list in payload.items()
        }

    def _store_ordered(
        self, data: dict[str, Any], order: dict[str, int], key: str, value: Any
    ) -> None:
        if key not in data:
            self._order_seq += 1
            order[key] = self._order_seq
        data[key] = value

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        self._init_lens_state()
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)
        prepared = self._prepare_steering({key: sv_list})
        self._store_ordered(self._steering_data, self._steering_order, key, prepared[key])
        self._bump_gen()

    def set_steering_data_many(self, pickled_data: bytes) -> int:
        """Store steering vectors for many keys in one RPC.

        ``pickled_data`` is ``pickle.dumps({key: list[SteeringVector]})``.
        Equivalent to one ``set_steering_data`` per key (the offline path sends
        one call per ``generate()`` instead of one per request).
        """
        self._init_lens_state()
        payload: dict[str, list[SteeringVector]] = pickle.loads(pickled_data)
        prepared = self._prepare_steering(payload)
        for key, vectors in prepared.items():
            self._store_ordered(self._steering_data, self._steering_order, key, vectors)
        self._bump_gen()
        return len(prepared)

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._init_lens_state()
        if self._steering_data.pop(key, None) is not None:
            self._steering_order.pop(key, None)
            self._bump_gen()

    def clear_steering_data_many(self, keys: list[str]) -> None:
        """Remove steering data for several keys in one RPC."""
        self._init_lens_state()
        removed = False
        for key in keys:
            if self._steering_data.pop(key, None) is not None:
                self._steering_order.pop(key, None)
                removed = True
        if removed:
            self._bump_gen()

    # ------------------------------------------------------------------
    # Captured activations
    # ------------------------------------------------------------------

    def _flush_capture_blocks(self, wait: bool = True) -> None:
        """File pending device->host copies per request, in pageable memory.

        ``wait=False`` files only the copies that have already landed (they
        complete in order) and leaves the rest pending.  Rows are copied out
        of the pinned staging block so the block returns to PyTorch's
        pinned-memory cache instead of staying page-locked until every
        request sharing it has been retrieved.
        """
        blocks = self._capture_blocks
        if not blocks:
            return
        done = 0
        for block in blocks:
            if block.event is not None:
                if wait:
                    block.event.synchronize()
                elif not block.event.query():
                    break
            # One pageable copy per block (not per request): the pinned block
            # goes back to the cache, requests keep cheap views of the copy.
            page = block.host.clone() if block.host.is_pinned() else block.host
            offset = 0
            for req_id, n in block.segments:
                layer_states = self._captured_states.setdefault(req_id, {})
                layer_states.setdefault(block.layer_idx, []).append(
                    page[offset : offset + n]
                )
                offset += n
            done += 1
        self._capture_blocks = blocks[done:]

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        self._init_lens_state()
        self._flush_capture_blocks()
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                logger.debug("Cleared leaked activations for %s", req_id)

    def _build_payload(
        self, internal_req_id: str
    ) -> dict[str, dict[str, "Float[torch.Tensor, 'n_layers total_pos hidden_dim']"]]:  # type: ignore[reportUndefinedVariable]
        """Materialise the stacked-tensor payload for one internal request id.

        Pops the entry from ``_captured_states`` so successive calls do not
        re-emit the same data. Shared by :meth:`get_captured_states` and
        :meth:`get_captured_states_batch`.
        """
        layer_dict = self._captured_states.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        chunks = [layer_dict[idx] for idx in sorted_indices]
        lengths = {sum(c.shape[0] for c in layer_chunks) for layer_chunks in chunks}
        first = chunks[0][0]
        if len(lengths) == 1 and all(c.device.type == "cpu" for lc in chunks for c in lc):
            # Write each layer straight into the stacked output (one copy).
            stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = torch.empty(  # type: ignore[reportUndefinedVariable]
                (len(chunks), lengths.pop(), *first.shape[1:]), dtype=first.dtype
            )
            for li, layer_chunks in enumerate(chunks):
                torch.cat(layer_chunks, dim=0, out=stacked[li])
        else:  # ragged layers: let torch.stack raise as upstream did
            stacked = torch.stack([torch.cat(c, dim=0) for c in chunks], dim=0)
        if stacked.device.type != "cpu":
            stacked = stacked.cpu()
        return {"activations": {"residual_stream": stacked}}

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle + zstd for safe ZMQ
        transport (the compression matters most when the response crosses
        the network in the OpenAI/Inspect HTTP path).

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        self._init_lens_state()
        self._flush_capture_blocks()
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                payload = self._build_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_states_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Retrieve captured activations for many requests in one RPC.

        Equivalent to calling :meth:`get_captured_states` once per id, but
        emits a single payload covering every request that has data. At
        large batch sizes the per-request ``collective_rpc`` roundtrip is
        the dominant cost on the offline ``LLM.generate`` path; batching it
        cuts N round-trips to one.

        Returns ``pickle.dumps({external_req_id: payload, ...})`` where
        each ``payload`` is ``{"activations": {"residual_stream": Tensor}}``.
        Missing or unmatched ids are simply absent; ``None`` if nothing
        matched at all.

        We deliberately don't ``zstd``-compress here: this RPC only fires
        on the offline ``LLM.generate`` path, which is in-process IPC,
        not HTTP. ``get_captured_states`` keeps zstd for the OpenAI/Inspect
        path where the response crosses the network.
        """
        self._init_lens_state()
        if not external_req_ids:
            return None
        t0 = time.perf_counter()
        self._flush_capture_blocks()
        t1 = time.perf_counter()
        out: dict[str, dict[str, Any]] = {}
        grouped = _by_external_id(list(self._captured_states), external_req_ids)
        for external_req_id, internal_ids in grouped.items():
            for req_id in internal_ids:
                payload = self._build_payload(req_id)
                acts = payload["activations"]
                acts["residual_stream"] = pack_tensor(acts["residual_stream"])
                out[external_req_id] = payload
        if not out:
            return None
        t2 = time.perf_counter()
        raw = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
        stats = self._lens_stats
        stats["retrieval_flush_s"] += t1 - t0
        stats["retrieval_build_s"] += t2 - t1
        stats["retrieval_pickle_s"] += time.perf_counter() - t2
        return raw

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        self._init_lens_state()
        self._flush_capture_blocks()
        return len(self._captured_states)

    def _debug_layer_discovery(self) -> list[int]:
        """Global indices of decoder layers found by discovery (testing)."""
        return sorted(_discover_layer_modules(self))

    # ------------------------------------------------------------------
    # Attention Q/K capture (called via collective_rpc)
    # ------------------------------------------------------------------

    def _select_qk_attention_modules(self) -> dict[int, Any]:
        """Global layer index → the decoder ``Attention`` module inside it.

        Reuses decoder-layer discovery (registry or ``VLLM_LENS_LAYER_PATH``,
        count-checked) so Q/K layer indices are exactly the residual-stream
        ones.  Layers without a softmax-attention wrapper (Mamba /
        linear-attention blocks in hybrids) are omitted; MLA is refused.
        """
        from vllm.model_executor.layers.attention import Attention, MLAAttention

        layer_map = _discover_layer_modules(self)
        selected: dict[int, Any] = {}
        for layer_idx, layer in layer_map.items():
            attn_modules = [
                m
                for m in layer.modules()
                if isinstance(m, (Attention, MLAAttention))
                and _is_decoder_mixer(m, (Attention, MLAAttention))
            ]
            if any(isinstance(m, MLAAttention) for m in attn_modules):
                raise RuntimeError(
                    "output_qk is not supported for MLA models (DeepSeek-style "
                    "multi-head latent attention): the Attention wrapper never "
                    "sees materialized per-head Q/K, so there is nothing to "
                    "capture. Residual-stream capture and steering still work."
                )
            if len(attn_modules) > 1:
                raise RuntimeError(
                    f"Decoder layer {layer_idx} contains {len(attn_modules)} "
                    "decoder Attention modules; output_qk expects exactly one."
                )
            if attn_modules:
                selected[layer_idx] = attn_modules[0]
        return selected

    # -- raw Q/K reference capture (testing only) ------------------------------
    #
    # Plain torch pre-hooks on the same ``Attention`` modules, recording the
    # *whole* batch's Q/K per forward with no per-request slicing, merging or
    # serialization.  Tests compare the production capture against this
    # bit-for-bit (``torch.equal``) to verify the plumbing exactly — hook
    # firing under chunked prefill, query_start_loc slicing, TP head-shard
    # order, KV-replication dedupe, PP layer concat, wire round-trip.

    def _debug_raw_qk_install(self, layer_indices: list[int]) -> list[int]:
        self._debug_raw_qk = {i: {"q": [], "k": []} for i in layer_indices}
        self._debug_raw_qk_handles = []
        selected = self._select_qk_attention_modules()
        for layer_idx in layer_indices:
            if layer_idx not in selected:  # not on this PP stage
                continue
            module = selected[layer_idx]

            def hook(_m, args, kwargs, *, _idx=layer_idx, _mod=module):
                query = kwargs.get("query", args[0] if args else None)
                key = kwargs.get("key", args[1] if len(args) > 1 else None)
                if query is None or key is None:
                    return None
                store = self._debug_raw_qk[_idx]
                store["q"].append(
                    query.reshape(-1, _mod.num_heads, _mod.head_size).cpu()
                )
                store["k"].append(
                    key.reshape(-1, _mod.num_kv_heads, _mod.head_size).cpu()
                )
                return None

            self._debug_raw_qk_handles.append(
                module.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return sorted(selected)

    def _debug_raw_qk_pop(self) -> bytes:
        """Return and clear the raw capture: pickled ``{layer: {q, k}}`` plus ranks."""
        from vllm.distributed import parallel_state as ps

        out: dict[str, Any] = {
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
            "pp_rank": int(ps.get_pp_group().rank_in_group),
            "layers": {},
        }
        for idx, st in self._debug_raw_qk.items():
            if st["q"]:
                out["layers"][idx] = {
                    "q": torch.cat(st["q"], dim=0),
                    "k": torch.cat(st["k"], dim=0),
                }
            st["q"].clear()
            st["k"].clear()
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_raw_qk_remove(self) -> None:
        for h in getattr(self, "_debug_raw_qk_handles", []):
            h.remove()
        self._debug_raw_qk_handles = []
        self._debug_raw_qk = {}

    def install_qk_hooks(self) -> None:
        """Register a pre-hook on every decoder ``Attention`` module. Idempotent.

        Separate from :meth:`install_hooks` so residual-stream/steering
        users pay no extra per-forward cost.  Installed — and capturing —
        on **every** TP rank, because ``Attention`` inputs are sharded by
        head (unlike the residual stream, which is replicated post
        all-reduce).

        Also records, per layer, the exact kernel parameters needed to
        replay the attention softmax offline (``vllm_lens.attention``):
        vLLM has already normalized model-specific conventions (custom
        scales, sliding-window forms, ...) into these values.
        """
        if self._qk_hooks_installed:
            return
        self._init_lens_state()

        selected = self._select_qk_attention_modules()

        if not selected:
            raise RuntimeError(
                "output_qk found no decoder attention layers to hook "
                "(is this an attention-free model?)."
            )

        # Reset to instance-level state (class-level defaults are shared).
        self._captured_qk = {}
        self._qk_layer_meta = {}
        self._install_pass_tracking()
        self._qk_hooks_installed = True

        model_config = getattr(self, "model_config", None)
        try:
            total_kv_heads = int(model_config.get_total_num_kv_heads())  # type: ignore[union-attr]
        except Exception:
            total_kv_heads = 0

        from vllm.distributed import parallel_state as ps

        tp_size = self.parallel_config.tensor_parallel_size
        self._qk_tp_rank = int(ps.get_tensor_model_parallel_rank())
        self._qk_pp_rank = int(ps.get_pp_group().rank_in_group)
        self._qk_tp_size = int(tp_size)

        for layer_idx, module in sorted(selected.items()):
            impl = module.impl
            self._qk_layer_meta[layer_idx] = {
                "layer_name": getattr(module, "layer_name", ""),
                "scale": float(getattr(impl, "scale", None) or module.head_size**-0.5),
                "logits_soft_cap": float(getattr(impl, "logits_soft_cap", None) or 0.0),
                "sliding_window": list(
                    _normalize_sliding_window(getattr(impl, "sliding_window", None))
                ),
                "alibi_slopes": _to_float_list(getattr(impl, "alibi_slopes", None)),
                "sinks": _to_float_list(getattr(impl, "sinks", None)),
                "use_alibi_sqrt": bool(getattr(module, "use_alibi_sqrt", False)),
                "num_heads_local": int(module.num_heads),
                "num_kv_heads_local": int(module.num_kv_heads),
                "head_size": int(module.head_size),
                "num_kv_heads_total": total_kv_heads or int(module.num_kv_heads),
            }
            module.register_forward_pre_hook(
                _make_qk_hook(self, layer_idx, module), with_kwargs=True
            )

    def _build_qk_payload(self, internal_req_id: str) -> dict[str, Any]:
        """Materialise the stacked Q/K payload for one internal request id.

        Pops the entry so successive calls do not re-emit the same data.
        Q stacks to ``(n_layers, total_pos, local_q_heads, head_size)`` and
        K to ``(n_layers, total_pos, local_kv_heads, head_size)``.
        """
        layer_dict = self._captured_qk.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        q_layers = [torch.cat(layer_dict[i]["q"], dim=0) for i in sorted_indices]
        k_layers = [torch.cat(layer_dict[i]["k"], dim=0) for i in sorted_indices]
        if len({t.shape for t in q_layers}) > 1 or len({t.shape for t in k_layers}) > 1:
            # Heterogeneous head counts across captured layers (exotic
            # hybrids). Stacking would silently misalign — refuse loudly.
            raise RuntimeError(
                "output_qk captured layers with heterogeneous Q/K shapes; "
                "capture a single layer at a time for this model."
            )
        return {
            "qk": {
                "q": torch.stack(q_layers, dim=0),
                "k": torch.stack(k_layers, dim=0),
                "layers": sorted_indices,
                "meta": [self._qk_layer_meta[i] for i in sorted_indices],
                "tp_rank": getattr(self, "_qk_tp_rank", 0),
                "pp_rank": getattr(self, "_qk_pp_rank", 0),
                "tp_size": getattr(self, "_qk_tp_size", 1),
            }
        }

    def get_captured_qk(self, external_req_id: str) -> bytes | None:
        """Retrieve captured Q/K for a request (zstd + pickle).

        Same ``"{external_req_id}-"`` prefix-match convention as
        :meth:`get_captured_states`.  Every TP rank returns its own head
        shard, tagged with its ranks; the plugin merges along the head
        dimension.  Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_qk):
            if req_id.startswith(prefix):
                payload = self._build_qk_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_qk_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Batched :meth:`get_captured_qk` for the offline path (plain pickle)."""
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        grouped = _by_external_id(list(self._captured_qk), external_req_ids)
        for external_req_id, internal_ids in grouped.items():
            for req_id in internal_ids:
                out[external_req_id] = self._build_qk_payload(req_id)
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def clear_captured_qk(self, external_req_id: str) -> None:
        """Remove captured Q/K without returning it (abort/disconnect cleanup)."""
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_qk):
            if req_id.startswith(prefix):
                del self._captured_qk[req_id]
                logger.debug("Cleared leaked Q/K capture for %s", req_id)

    def _debug_captured_qk_count(self) -> int:
        """Return the number of entries in _captured_qk (for testing)."""
        return len(self._captured_qk)

    # ------------------------------------------------------------------
    # Hook data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def _validate_hooks(self, hooks: list[Hook]) -> None:
        num_layers = _get_total_num_layers(self)
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )

    def set_hook_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store hook definitions for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``Hook`` instances (using cloudpickle for the callable
        ``fn``), validates layer indices against the model, and stores them
        keyed by *key* (an external request ID or ``_hook_id`` sentinel).
        """
        self._init_lens_state()
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        self._validate_hooks(hooks)
        self._store_ordered(self._hook_data, self._hook_order, key, hooks)
        self._bump_gen()

    def set_hook_data_many(self, pickled_data: bytes) -> int:
        """Store hook definitions for many keys in one RPC.

        ``pickled_data`` is ``cloudpickle.dumps({key: list[Hook]})``.
        """
        self._init_lens_state()
        payload: dict[str, list[Hook]] = cloudpickle.loads(pickled_data)
        for hooks in payload.values():
            self._validate_hooks(hooks)
        for key, hooks in payload.items():
            self._store_ordered(self._hook_data, self._hook_order, key, hooks)
        self._bump_gen()
        return len(payload)

    def get_hook_results(self, external_req_id: str) -> bytes | None:
        """Retrieve hook results (``ctx.saved`` dicts) for a request.

        Returns from ALL ranks (including PP ranks that own different
        layers).  The plugin merges results across ranks.
        Matches by ``"{external_req_id}-"`` prefix on ``_hook_contexts``.
        Returns ``{str(hook_position): ctx.saved}`` pickled, where
        ``hook_position`` indexes the per-request hook list.
        """
        self._init_lens_state()
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(pos): ctx.saved for pos, ctx in contexts.items()}
                return pickle.dumps(saved_dicts)
        return None

    def get_hook_results_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Batched :meth:`get_hook_results`: ``{external_id: {pos: saved}}`` pickled."""
        self._init_lens_state()
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        grouped = _by_external_id(list(self._hook_contexts), external_req_ids)
        for external_req_id, internal_ids in grouped.items():
            contexts = self._hook_contexts.pop(internal_ids[0])
            out[external_req_id] = {str(pos): ctx.saved for pos, ctx in contexts.items()}
        if not out:
            return None
        return pickle.dumps(out)

    def clear_hook_data(self, key: str) -> None:
        """Remove hook definitions for a completed request."""
        self._init_lens_state()
        if self._hook_data.pop(key, None) is not None:
            self._hook_order.pop(key, None)
            self._bump_gen()

    def clear_hook_data_many(self, keys: list[str]) -> None:
        """Remove hook definitions for several keys in one RPC."""
        self._init_lens_state()
        removed = False
        for key in keys:
            if self._hook_data.pop(key, None) is not None:
                self._hook_order.pop(key, None)
                removed = True
        if removed:
            self._bump_gen()

    def clear_hook_contexts(self, external_req_id: str) -> None:
        """Remove hook contexts for a completed or aborted request.

        Prefix-match cleanup, same pattern as ``clear_captured_states``.
        """
        self._init_lens_state()
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                del self._hook_contexts[req_id]

    def clear_hook_contexts_many(self, external_req_ids: list[str]) -> None:
        """Batched :meth:`clear_hook_contexts`."""
        self._init_lens_state()
        grouped = _by_external_id(list(self._hook_contexts), external_req_ids)
        for internal_ids in grouped.values():
            for req_id in internal_ids:
                self._hook_contexts.pop(req_id, None)

    # ------------------------------------------------------------------
    # Persistent hook management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_persistent_hooks(self, pickled_data: bytes) -> None:
        """Append hooks that apply to every subsequent request.

        Accepts cloudpickle'd ``list[Hook]``.  Validates layer indices.
        Appends to existing persistent hooks (call ``clear_persistent_hooks``
        first for a clean slate).  Also ensures forward hooks are installed
        on the model layers.
        """
        self.install_hooks()
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        self._validate_hooks(hooks)
        self._persistent_hooks.extend(hooks)
        self._refresh_persistent_layers()
        self._bump_gen()

    def get_all_hook_results(self) -> bytes | None:
        """Retrieve accumulated persistent hook contexts from all requests.

        Returns from ALL ranks (for PP support).  Does NOT clear — call
        ``clear_persistent_hooks`` explicitly.

        Returns pickled ``{internal_req_id: {hook_idx_str: ctx.saved}}``.
        """
        self._init_lens_state()
        if not self._persistent_hook_contexts:
            return None
        results: dict[str, dict[str, dict[str, Any]]] = {}
        for req_id, contexts in self._persistent_hook_contexts.items():
            results[req_id] = {str(pos): ctx.saved for pos, ctx in contexts.items()}
        return pickle.dumps(results)

    def clear_persistent_hooks(self) -> None:
        """Remove persistent hooks and all accumulated contexts."""
        self._init_lens_state()
        self._persistent_hooks = []
        self._persistent_hook_contexts = {}
        self._refresh_persistent_layers()
        self._bump_gen()

    def clear_persistent_hook_results(self) -> None:
        """Drop accumulated persistent-hook contexts, keeping hooks registered.

        ``get_all_hook_results`` never drains, so results pile up across
        requests. This clears them without unregistering the hooks (so a
        fitted lens does not need re-uploading), letting a client bound
        accumulation by clearing between turns.
        """
        self._init_lens_state()
        self._persistent_hook_contexts = {}

    # ------------------------------------------------------------------
    # Parameter prefetch (called via collective_rpc — all ranks in sync)
    # ------------------------------------------------------------------

    _prefetched_params: dict[str, torch.Tensor] = {}

    def prefetch_parameters(self, names: list[str]) -> None:
        """Pre-fetch and gather parameters across TP and PP ranks.

        Safe to call PP collectives here because ``collective_rpc``
        runs on all ranks simultaneously.  Results are stored in
        ``_prefetched_params`` for use by ``HookContext.get_parameter``.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self._init_lens_state()
        from vllm.model_executor.models.utils import PPMissingLayer

        model = _runner_model(self.model_runner)
        tp_group = get_tp_group()
        pp_group = get_pp_group()

        for name in names:
            # Traverse to find the parameter.
            obj: Any = model
            parts = name.split(".")
            is_local = True
            for attr in parts:
                obj = getattr(obj, attr)
                if isinstance(obj, PPMissingLayer):
                    is_local = False
                    break

            param: torch.Tensor | None = None
            if is_local:
                local_t = torch.as_tensor(obj)

                # TP gather if sharded; otherwise reuse the existing tensor.
                module: Any = model
                for attr in parts[:-1]:
                    module = getattr(module, attr)
                tp_size = getattr(module, "tp_size", 1)
                if tp_size > 1:
                    gathered = [torch.empty_like(local_t) for _ in range(tp_size)]
                    dist.all_gather(gathered, local_t, group=tp_group.device_group)
                    gather_dim = getattr(module, "gather_dim", 0)
                    param = torch.cat(gathered, dim=gather_dim)
                else:
                    param = local_t  # no copy — reference to existing parameter

            # PP broadcast — safe here because all ranks are in this RPC.
            if pp_group.world_size > 1:
                has_it = torch.tensor(
                    [1 if is_local else 0], device="cuda", dtype=torch.int32
                )
                all_has = [torch.zeros_like(has_it) for _ in range(pp_group.world_size)]
                dist.all_gather(all_has, has_it, group=pp_group.device_group)
                source_pp = next(i for i, t in enumerate(all_has) if t.item() == 1)
                source_global = pp_group.ranks[source_pp]

                if param is None:
                    # Receive shape + dtype.
                    meta = torch.zeros(3, device="cuda", dtype=torch.int64)
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    ndim = int(meta[0].item())
                    dtype = _DTYPE_LIST[int(meta[1].item())]
                    shape_t = torch.zeros(ndim, device="cuda", dtype=torch.int64)
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )
                    shape = tuple(int(s) for s in shape_t.tolist())
                    param = torch.empty(shape, device="cuda", dtype=dtype)
                else:
                    meta = torch.tensor(
                        [param.ndim, _dtype_to_idx(param.dtype), 0],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    shape_t = torch.tensor(
                        list(param.shape),
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )

                dist.broadcast(param, src=source_global, group=pp_group.device_group)

            assert param is not None, f"Parameter {name!r} not found on any rank"
            self._prefetched_params[name] = param

    def clear_prefetched_params(self) -> None:
        """Remove all pre-fetched parameters."""
        self._init_lens_state()
        self._prefetched_params = {}
