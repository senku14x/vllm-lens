"""Regression tests for the per-pass plan: positions, pre-hooks, CUDA graphs.

* Position-specific steering must land on absolute positions during decode
  and across chunked-prefill chunks (the hook once treated every chunk as
  starting at position 0).
* Pre-hooks must run on models whose decoder layers are called by keyword
  (Qwen3-Next / Qwen3.5 / Qwen3.6).
* With CUDA graphs on, passes with hook work run eagerly (same results as
  an eager engine) and passes without it replay graphs.
"""

import gc
import os

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from vllm_lens import Hook, SteeringVector

from .conftest import MODEL_NAME

GPU_UTIL = float(os.environ.get("VLLM_LENS_TEST_GPU_UTIL", "0.3"))
HYBRID_MODEL = os.environ.get("VLLM_LENS_TEST_HYBRID_MODEL", "Qwen/Qwen3.5-0.8B")
PROMPT_TEXT = "The capital of France is Paris. The capital of Germany is"


def _engine(model=MODEL_NAME, **kwargs):
    return LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=2048,
        seed=0,
        kernel_config={"enable_jit_warmup": False},
        **kwargs,
    )


def _shutdown(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_graphs():
    os.environ.pop("VLLM_LENS_CUDA_GRAPHS", None)
    llm = _engine()
    yield llm
    _shutdown(llm)


def _capture(llm, ids, layer, max_tokens, steering=None, hooks=None):
    extra = {"output_residual_stream": [layer]}
    if steering is not None:
        extra["apply_steering_vectors"] = steering
    if hooks is not None:
        extra["apply_hooks"] = hooks
    sp = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True, extra_args=extra
    )
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)[0]
    return out.activations["residual_stream"][0].float(), list(out.outputs[0].token_ids)


def _changed_positions(base, steered):
    n = min(base.shape[0], steered.shape[0])
    d = (steered[:n] - base[:n]).norm(dim=-1)
    tol = 1e-3 * base[:n].norm(dim=-1).clamp_min(1.0)
    return torch.nonzero(d > tol).flatten().tolist()


def _ids(llm, text=PROMPT_TEXT):
    return llm.get_tokenizer()(text, add_special_tokens=False)["input_ids"]


def _sizes(llm):
    cfg = llm.llm_engine.model_config
    return cfg.hf_text_config.num_hidden_layers, cfg.get_hidden_size()


def _sv(vec, layer, pos):
    return [
        SteeringVector(
            activations=vec[None, None, :].clone(),
            layer_indices=[layer],
            position_indices=[pos],
        )
    ]


class TestDecodePositions:
    def test_position_zero_vector_not_reapplied_during_decode(self, llm_graphs):
        """A last-layer edit at position 0 reaches no other position."""
        n_layers, hidden = _sizes(llm_graphs)
        last = n_layers - 1
        ids = _ids(llm_graphs)
        torch.manual_seed(0)
        vec = torch.randn(hidden) * 4.0
        base, base_tok = _capture(llm_graphs, ids, last, 8)
        st, st_tok = _capture(llm_graphs, ids, last, 8, steering=_sv(vec, last, 0))
        assert _changed_positions(base, st) == [0]
        assert st_tok == base_tok

    def test_vector_at_generated_position_is_applied(self, llm_graphs):
        n_layers, hidden = _sizes(llm_graphs)
        layer = 2
        ids = _ids(llm_graphs)
        target = len(ids) + 2  # third generated position
        torch.manual_seed(1)
        vec = torch.randn(hidden) * 4.0
        base, _ = _capture(llm_graphs, ids, layer, 8)
        st, _ = _capture(llm_graphs, ids, layer, 8, steering=_sv(vec, layer, target))
        changed = _changed_positions(base, st)
        assert changed and changed[0] == target
        delta = st[target] - base[target]
        cos = torch.nn.functional.cosine_similarity(delta, vec, dim=0).item()
        assert cos > 0.99
        assert abs(delta.norm().item() / vec.norm().item() - 1) < 0.02


    def test_reused_sampling_params_keep_steering(self, llm_graphs):
        """The plugin must not strip steering out of the caller's SamplingParams."""
        _, hidden = _sizes(llm_graphs)
        ids = _ids(llm_graphs)
        torch.manual_seed(5)
        vec = torch.randn(hidden) * 4.0
        base, _ = _capture(llm_graphs, ids, 2, 1)
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "output_residual_stream": [2],
                "apply_steering_vectors": _sv(vec, 2, 1),
            },
        )
        for _ in range(2):
            out = llm_graphs.generate(
                [TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False
            )[0]
            acts = out.activations["residual_stream"][0].float()
            assert _changed_positions(base, acts) == [1]
        assert "apply_steering_vectors" in sp.extra_args


class TestChunkedPrefillPositions:
    @pytest.fixture(scope="class")
    def llm_chunked(self):
        llm = _engine(
            max_num_batched_tokens=64,
            max_num_seqs=16,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=64,
        )
        yield llm
        _shutdown(llm)

    @pytest.mark.parametrize("pos", [10, 74, 140])
    def test_vector_lands_on_its_position(self, llm_chunked, pos):
        n_layers, hidden = _sizes(llm_chunked)
        last = n_layers - 1
        ids = _ids(llm_chunked, " ".join(f"word{i}" for i in range(400)))[:187]
        torch.manual_seed(pos)
        vec = torch.randn(hidden) * 4.0
        base, _ = _capture(llm_chunked, ids, last, 1)
        st, _ = _capture(llm_chunked, ids, last, 1, steering=_sv(vec, last, pos))
        assert _changed_positions(base, st) == [pos]


class TestCudaGraphs:
    def test_graphs_on_and_only_hooked_passes_eager(self, llm_graphs):
        llm = llm_graphs
        cfg = llm.llm_engine.vllm_config
        assert not cfg.model_config.enforce_eager
        assert cfg.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
        n_layers, hidden = _sizes(llm)
        ids = _ids(llm)
        _capture(llm, ids, 1, 1)  # installs the hooks
        llm.collective_rpc("lens_stats", args=(True,))
        plain = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        llm.generate([TokensPrompt(prompt_token_ids=ids)] * 4, plain, use_tqdm=False)
        stats = llm.collective_rpc("lens_stats", args=(True,))[0]
        assert stats["cuda_graphs"] and stats["passes_forced_eager"] == 0
        # A broadcast vector touches every decode step: all passes run eagerly.
        sv = [SteeringVector(activations=torch.randn(1, hidden) * 0.1, layer_indices=[1])]
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=16,
            ignore_eos=True,
            extra_args={"apply_steering_vectors": sv},
        )
        llm.generate([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)
        stats = llm.collective_rpc("lens_stats", args=(True,))[0]
        assert stats["passes_forced_eager"] >= 15
        assert stats["hook_errors"] == 0

    def test_graph_engine_matches_eager_engine(self, llm_graphs):
        """Steering + capture through decode give the same tokens and states."""
        n_layers, hidden = _sizes(llm_graphs)
        ids = _ids(llm_graphs)
        torch.manual_seed(3)
        sv2d = [
            SteeringVector(
                activations=torch.randn(1, hidden), layer_indices=[3], norm_match=True, scale=0.5
            )
        ]
        vec = torch.randn(hidden) * 4.0
        cases = [
            dict(steering=sv2d),
            dict(steering=_sv(vec, 3, len(ids) + 1)),
        ]
        got = [_capture(llm_graphs, ids, 5, 12, **c) for c in cases]
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "0"
        try:
            eager = _engine()
            assert eager.llm_engine.vllm_config.model_config.enforce_eager
            ref = [_capture(eager, ids, 5, 12, **c) for c in cases]
            _shutdown(eager)
        finally:
            os.environ.pop("VLLM_LENS_CUDA_GRAPHS", None)
        for (acts, toks), (racts, rtoks) in zip(got, ref):
            assert toks == rtoks
            assert torch.equal(acts, racts)


class TestKeywordCalledLayers:
    """Qwen3.5 decoder layers are called as layer(positions=..., hidden_states=...)."""

    @pytest.fixture(scope="class")
    def llm_hybrid(self):
        llm = _engine(model=HYBRID_MODEL, language_model_only=True)
        yield llm
        _shutdown(llm)

    def test_pre_hook_runs_and_replaces_input(self, llm_hybrid):
        ids = _ids(llm_hybrid, "The quick brown fox jumps over the lazy")

        def count(ctx, h):
            ctx.saved["calls"] = ctx.saved.get("calls", 0) + 1
            ctx.saved["rows"] = int(h.shape[0])

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=4,
            ignore_eos=True,
            extra_args={"apply_hooks": [Hook(fn=count, layer_indices=[0, 5], pre=True)]},
        )
        out = llm_hybrid.generate([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)[0]
        saved = out.hook_results["0"]
        # layers 0 and 5, once for the prompt and once per decode step (the
        # scheduler may run one extra pass after the last token)
        assert saved["calls"] in (2 * 4, 2 * 5)

        def zero(ctx, h):
            return torch.zeros_like(h)

        base, _ = _capture(llm_hybrid, ids, 0, 1)
        hooked, _ = _capture(
            llm_hybrid, ids, 0, 1, hooks=[Hook(fn=zero, layer_indices=[0], pre=True)]
        )
        assert (hooked - base).abs().max().item() > 1e-3
        stats = llm_hybrid.collective_rpc("lens_stats")[0]
        assert stats["hook_errors"] == 0

    def test_pre_hook_after_layer0_sees_full_stream(self, llm_hybrid):
        """Pre-hook input at layer L == post-hook output at layer L-1."""
        ids = _ids(llm_hybrid)

        def grab_pre(ctx, h):
            ctx.saved.setdefault("pre", []).append(h.float().cpu())

        def grab_post(ctx, h):
            ctx.saved.setdefault("post", []).append(h.float().cpu())

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "apply_hooks": [
                    Hook(fn=grab_post, layer_indices=[4]),
                    Hook(fn=grab_pre, layer_indices=[5], pre=True),
                ]
            },
        )
        out = llm_hybrid.generate([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)[0]
        post = out.hook_results["0"]["post"][0]
        pre = out.hook_results["1"]["pre"][0]
        assert torch.equal(pre, post)
