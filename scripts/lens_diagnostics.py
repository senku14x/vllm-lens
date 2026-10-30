#!/usr/bin/env python
"""Known-answer checks and a hook-overhead benchmark for vllm-lens.

Engine startup dominates run time, so ``suite`` runs the per-engine checks
on one engine; ``chunked`` needs its own engine configuration.  Each check
prints one ``RESULT`` JSON line, which makes runs against two checkouts easy
to compare (put the checkout to test first on ``PYTHONPATH``).

  suite      position + prehook + parity (one engine, HF reference after it)
  position   3-D steering lands on the requested absolute positions during
             decode.
  chunked    ... and across chunked-prefill chunks.
  prehook    pre-hooks run on decoder layers that vLLM calls by keyword
             (Qwen3.5 / Qwen3.6 / Qwen3-Next).
  parity     captured residual streams match a Hugging Face forward pass.
  bench      wall time of generate() with and without per-request
             interventions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch


def _llm(args, **overrides):
    from vllm import LLM

    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=0,
    )
    if args.kv_cache_gb:
        # A fixed KV-cache size skips vLLM's free-memory profiling, which
        # fails when other processes on the GPU allocate or free meanwhile.
        kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gb * 2**30)
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.eager:
        kwargs["enforce_eager"] = True
    if not args.jit_warmup:
        # vLLM 0.30 compiles ~120 Triton kernel variants at startup; compiling
        # the few a small test needs on first use is much faster.
        kwargs["kernel_config"] = {"enable_jit_warmup": False}
    kwargs.update(overrides)
    return LLM(**kwargs)


def _shutdown(llm) -> None:
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 - best effort; the engine may be gone
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _result(name: str, **fields) -> None:
    print("RESULT " + json.dumps({"check": name, **fields}), flush=True)


def _stats(llm, reset=False):
    try:
        return llm.collective_rpc("lens_stats", args=(reset,))[0]
    except Exception as e:  # upstream has no lens_stats
        return {"unavailable": type(e).__name__}


def _sizes(llm) -> tuple[int, int]:
    cfg = llm.llm_engine.model_config
    return int(cfg.hf_text_config.num_hidden_layers), int(cfg.get_hidden_size())


def _prompt_ids(llm, text: str) -> list[int]:
    return llm.get_tokenizer()(text, add_special_tokens=False)["input_ids"]


def _run(llm, prompt_ids, max_tokens, layer, steering=None, extra=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ea = {"output_residual_stream": [layer]}
    if steering is not None:
        ea["apply_steering_vectors"] = steering
    if extra:
        ea.update(extra)
    sp = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True, extra_args=ea
    )
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], [sp], use_tqdm=False)[0]
    return out.activations["residual_stream"][0].float(), list(out.outputs[0].token_ids)


def _delta_report(base, steered, vec, positions_expected):
    """Per-position comparison of a steered capture against a clean one."""
    n = min(base.shape[0], steered.shape[0])
    d = (steered[:n] - base[:n]).norm(dim=-1)
    tol = 1e-3 * base[:n].norm(dim=-1).clamp_min(1.0)
    changed = [int(i) for i in torch.nonzero(d > tol).flatten().tolist()]
    hits = {}
    for p in positions_expected:
        if p < n:
            delta = steered[p] - base[p]
            hits[p] = {
                "norm_ratio": round(float(delta.norm() / vec.norm()), 4),
                "cos": round(float(torch.nn.functional.cosine_similarity(delta, vec, dim=0)), 4),
            }
    return n, changed, hits


def _sv(vec, layer, pos):
    from vllm_lens import SteeringVector

    return [
        SteeringVector(
            activations=vec[None, None, :].clone(), layer_indices=[layer], position_indices=[pos]
        )
    ]


def check_position(llm) -> None:
    n_layers, hidden = _sizes(llm)
    last = n_layers - 1
    torch.manual_seed(0)
    ids = _prompt_ids(llm, "The capital of France is Paris. The capital of Germany is")
    n_prompt = len(ids)
    gen = 8

    # (A) Steer the LAST layer at position 0 and generate. A last-layer edit at
    # position 0 cannot reach any other position, so only {0} may change.
    vec = torch.randn(hidden) * 4.0
    base, base_tok = _run(llm, ids, gen, last)
    st, st_tok = _run(llm, ids, gen, last, steering=_sv(vec, last, 0))
    n, changed, hits = _delta_report(base, st, vec, [0])
    _result(
        "position_A_last_layer_pos0_with_decode",
        captured_rows=n,
        expected_rows=n_prompt + gen - 1,
        changed=changed,
        hits=hits,
        same_tokens=base_tok == st_tok,
        ok=changed == [0] and n == n_prompt + gen - 1,
    )

    # (B) Steer layer L at a GENERATED position. The edit must appear exactly
    # there in layer L's output.
    layer = min(2, last)
    target = n_prompt + 2
    vec_b = torch.randn(hidden) * 4.0
    base_b, _ = _run(llm, ids, gen, layer)
    st_b, _ = _run(llm, ids, gen, layer, steering=_sv(vec_b, layer, target))
    n, changed_b, hits_b = _delta_report(base_b, st_b, vec_b, [target])
    first = changed_b[0] if changed_b else None
    _result(
        "position_B_generated_position",
        target=target,
        captured_rows=n,
        first_changed=first,
        hits=hits_b,
        ok=first == target and abs(hits_b.get(target, {}).get("cos", 0) - 1) < 0.02,
    )


def check_prehook(llm) -> None:
    from vllm import SamplingParams

    from vllm_lens import Hook

    prompt = "The quick brown fox jumps over the lazy"
    n_layers, _ = _sizes(llm)
    mid = min(5, n_layers - 1)

    def count_calls(ctx, h):
        ctx.saved["calls"] = ctx.saved.get("calls", 0) + 1
        return None

    def zero_input(ctx, h):
        return torch.zeros_like(h)

    def grab_pre(ctx, h):
        ctx.saved.setdefault("pre", []).append(h.float().cpu())

    def grab_post(ctx, h):
        ctx.saved.setdefault("post", []).append(h.float().cpu())

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        extra_args={"apply_hooks": [Hook(fn=count_calls, layer_indices=[0], pre=True)]},
    )
    out = llm.generate([prompt], [sp], use_tqdm=False)[0]
    calls = ((getattr(out, "hook_results", None) or {}).get("0") or {}).get("calls")

    cap = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [0]})
    base = llm.generate([prompt], [cap], use_tqdm=False)[0].activations["residual_stream"][0].float()
    hooked_sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={
            "output_residual_stream": [0],
            "apply_hooks": [Hook(fn=zero_input, layer_indices=[0], pre=True)],
        },
    )
    hooked = llm.generate([prompt], [hooked_sp], use_tqdm=False)[0].activations["residual_stream"][0].float()
    diff = float((hooked - base).abs().max())

    # Pre-hook input at layer L must equal post-hook output at layer L-1.
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={
            "apply_hooks": [
                Hook(fn=grab_post, layer_indices=[mid - 1]),
                Hook(fn=grab_pre, layer_indices=[mid], pre=True),
            ]
        },
    )
    out = llm.generate([prompt], [sp], use_tqdm=False)[0]
    hr = getattr(out, "hook_results", None) or {}
    try:
        pre_eq_post = bool(torch.equal(hr["1"]["pre"][0], hr["0"]["post"][0]))
    except (KeyError, IndexError):
        pre_eq_post = None
    _result(
        "prehook",
        calls_layer0_4_tokens=calls,
        zeroing_changes_capture=diff > 1e-3,
        pre_at_L_equals_post_at_L_minus_1=pre_eq_post,
        ok=bool(calls) and calls >= 4 and diff > 1e-3 and pre_eq_post is True,
    )


PARITY_PROMPTS = [
    "The future of AI is",
    "Once upon a time in a land far away, there lived a",
    "def fibonacci(n):\n    ",
    "Scientists recently discovered that the universe",
]


def capture_for_parity(llm):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = llm.get_tokenizer()
    n_layers, _ = _sizes(llm)
    layers = sorted({0, n_layers // 2, n_layers - 2})
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in PARITY_PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": layers})
    outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], [sp] * len(ids), use_tqdm=False)
    return ids, layers, [o.activations["residual_stream"].float() for o in outs]


def _text_model_from_multimodal(path: str, dtype: torch.dtype):
    """The language model of an image-text checkpoint, loaded strictly."""
    import glob

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path).text_config
    hf = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    state = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        for k, v in load_file(f).items():
            if k.startswith("model.language_model."):
                state["model." + k[len("model.language_model."):]] = v
            elif k.startswith("lm_head."):
                state[k] = v
    missing, unexpected = hf.load_state_dict(state, strict=False)
    missing = [k for k in missing if not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    if cfg.tie_word_embeddings:
        hf.tie_weights()
    return hf.to("cuda").eval()


def check_parity(model, dtype, ids, layers, got) -> None:
    from transformers import AutoModelForCausalLM

    kw = dict(dtype=getattr(torch, dtype), device_map="cuda")
    try:
        hf = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
    except (AttributeError, ValueError):  # e.g. Qwen3.5 image-text checkpoints
        hf = _text_model_from_multimodal(model, getattr(torch, dtype))
    worst = {}
    with torch.no_grad():
        for x, g in zip(ids, got):
            hs = hf(input_ids=torch.tensor([x], device="cuda"), output_hidden_states=True).hidden_states
            for li, L in enumerate(layers):
                # HF hidden_states[L + 1] is the output of decoder layer L.
                ref = hs[L + 1][0].float().cpu()
                cos = torch.nn.functional.cosine_similarity(g[li], ref, dim=-1).min().item()
                worst[L] = round(min(worst.get(L, 1.0), cos), 6)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    _result("parity_vs_hf", layers=layers, min_cos=worst, ok=all(v > 0.99 for v in worst.values()))


def cmd_suite(args) -> None:
    llm = _llm(args)
    check_position(llm)
    check_prehook(llm)
    ids, layers, got = capture_for_parity(llm)
    _result("lens_stats", stats=_stats(llm))
    _shutdown(llm)
    check_parity(args.model, args.dtype, ids, layers, got)


def cmd_position(args) -> None:
    llm = _llm(args)
    check_position(llm)
    _result("lens_stats", stats=_stats(llm))


def cmd_prehook(args) -> None:
    llm = _llm(args)
    check_prehook(llm)
    _result("lens_stats", stats=_stats(llm))


def cmd_parity(args) -> None:
    llm = _llm(args)
    ids, layers, got = capture_for_parity(llm)
    _shutdown(llm)
    check_parity(args.model, args.dtype, ids, layers, got)


def cmd_chunked(args) -> None:
    chunk = args.chunk
    llm = _llm(
        args,
        max_num_batched_tokens=chunk,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=chunk,
    )
    n_layers, hidden = _sizes(llm)
    last = n_layers - 1
    torch.manual_seed(1)
    text = " ".join(f"word{i}" for i in range(400))
    ids = _prompt_ids(llm, text)[: 3 * chunk - 5]
    vec = torch.randn(hidden) * 4.0
    base, _ = _run(llm, ids, 1, last)
    results = {}
    for pos in (10, chunk + 10, 2 * chunk + 10):
        st, _ = _run(llm, ids, 1, last, steering=_sv(vec, last, pos))
        n, changed, hits = _delta_report(base, st, vec, [pos])
        results[pos] = {"changed": changed, "hits": hits, "ok": changed == [pos]}
    _result(
        "position_C_chunked_prefill",
        n_prompt=len(ids),
        chunk=chunk,
        results=results,
        ok=all(r["ok"] for r in results.values()),
    )
    _result("lens_stats", stats=_stats(llm))


def cmd_bench(args) -> None:
    from vllm import SamplingParams

    from vllm_lens import SteeringVector

    llm = _llm(args)
    n_layers, hidden = _sizes(llm)
    B, T = args.batch, args.new_tokens
    prompts = [f"Write a short story about a robot named R{i} who" for i in range(B)]
    layer = n_layers // 2

    def timed(label, make_sps, reps=args.reps):
        # Fresh SamplingParams every repetition: a reused object must not be
        # what is measured (it used to lose its steering after one call).
        times = []
        for _ in range(reps):
            sps = make_sps()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sps, use_tqdm=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        _result("bench", case=label, batch=B, new_tokens=T, best_s=round(min(times), 3), all_s=[round(t, 3) for t in times], stats=_stats(llm, reset=True))

    def plain():
        return SamplingParams(temperature=0.0, max_tokens=T, ignore_eos=True)

    llm.generate(prompts[:8], plain(), use_tqdm=False)  # warm-up
    timed("plain", plain)
    # A capture request installs the hooks; plain generation afterwards measures
    # the cost of installed-but-idle hooks.
    llm.generate(prompts[:1], SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [layer]}), use_tqdm=False)
    _stats(llm, reset=True)
    timed("plain_after_hooks_installed", plain)

    torch.manual_seed(0)
    vec2d = [torch.randn(1, hidden) * 0.01 for _ in range(B)]
    vec3d = [torch.randn(1, 1, hidden) * 0.01 for _ in range(B)]

    def steer2d():
        # One distinct broadcast (2-D) vector per request: applies on every step.
        return [
            SamplingParams(
                temperature=0.0,
                max_tokens=T,
                ignore_eos=True,
                extra_args={"apply_steering_vectors": [SteeringVector(activations=v, layer_indices=[layer])]},
            )
            for v in vec2d
        ]

    def steer3d():
        # Activation-oracle style: one position-specific vector on a prompt token.
        return [
            SamplingParams(
                temperature=0.0,
                max_tokens=T,
                ignore_eos=True,
                extra_args={
                    "apply_steering_vectors": [
                        SteeringVector(activations=v, layer_indices=[layer], position_indices=[3])
                    ]
                },
            )
            for v in vec3d
        ]

    def capture():
        # Prefill-only capture of three layers at every prompt position.
        return SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [1, layer, n_layers - 2]})

    timed("steer_2d_per_request", steer2d)
    timed("steer_3d_prompt_position", steer3d)
    timed("capture_prefill_3_layers", capture)
    _result("lens_stats", stats=_stats(llm))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["suite", "position", "chunked", "prehook", "parity", "bench"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-util", type=float, default=0.12)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--kv-cache-gb", type=float, default=4.0, help="0 = profile free memory")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--eager", action="store_true", help="pass enforce_eager=True")
    p.add_argument("--jit-warmup", action="store_true", help="keep vLLM's startup JIT warmup")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    import vllm_lens

    print(f"vllm_lens from {os.path.dirname(vllm_lens.__file__)}", flush=True)
    {
        "suite": cmd_suite,
        "position": cmd_position,
        "chunked": cmd_chunked,
        "prehook": cmd_prehook,
        "parity": cmd_parity,
        "bench": cmd_bench,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
