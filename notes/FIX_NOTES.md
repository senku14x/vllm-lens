# vllm-lens fix — notes (24–25 Oct 2026)

What was wrong with vllm-lens 1.3.0 (UKGovernmentBEIS, commit `3d11fb0`, vLLM 0.30.0), what was changed, how it was checked, and what is still open. The code is in this repo (`/workspace/vllm-lens-fixed`, uncommitted changes on branch `fix/hot-path-qwen36`). Neither the project's runtime copy (`/root/venvs/vllm-lens-src`, `/root/venvs/vl`) nor anything in `Neel Sprint/` was edited.

Raw evidence in this folder:
- `run_results.tsv`: every `RESULT` line from the diagnostic runs (file → JSON). `up_*` = unmodified upstream, `fx_*` = fixed.
- `gpu_tests_partial.tsv`: the GPU pytest run, as far as it got.
- `code_review_raw.txt`: the 10-agent independent review (5 reviewers + 5 adversarial verifiers), full output.

## Environment

- Test venv: `/root/venvs/vlfix` (Python 3.12, vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.20.2, same stack as the project's `vl` venv), with this repo installed editable. `/root` is container disk: it does not survive the instance.
- Baselines: an untouched checkout of upstream at `/root/vllm-lens-upstream` (git worktree of `3d11fb0`), put first on `PYTHONPATH`.
- Models: `Qwen/Qwen2.5-0.5B-Instruct` (dense, upstream's own test model) and `Qwen/Qwen3.5-0.8B` (same hybrid GatedDeltaNet architecture and vLLM model class as Qwen3.6-27B), copied to `/root/models_small/`.
- Known-answer checks: `scripts/lens_diagnostics.py` (`suite`, `position`, `chunked`, `prehook`, `parity`, `bench`). Each prints one `RESULT` JSON line per check.
- GPU was shared with the executor's jobs for most of the session and the CPU load average was ~25–45: timings are noisy (see "Benchmarks").

## Bugs found (all reproduced on unmodified upstream)

### 1. Position-specific steering lands on the wrong tokens (critical)
- **Cause:** `_hook_inner` computed each request's absolute start as `seq_lens[i] - n_query`, reading `seq_lens` with `getattr(attn_metadata, "seq_lens", None)`. In vLLM V1 `attn_metadata` is a *dict* of per-layer metadata, so this is always `None` and the code fell back to `abs_start = 0` for every chunk. Upstream's tests only steer with `max_tokens=1`, where 0 happens to be right.
- **Symptoms (measured):**
  - (A) a 3-D vector for position 0 was re-added to every generated token: changed positions `[0, 12, 13, …, 18]` instead of `[0]`, and the generated tokens changed. Seen on Qwen2.5 and Qwen3.5.
  - (B) a vector aimed at a generated position was never applied (delta 0).
  - (C) chunked prefill (64-token chunks, 187-token prompt): the position-10 vector was applied at 10, 74 and 138 (chunk-relative row 10 of every chunk); the vectors for 74 and 138 were never applied.
- **Fix:** absolute starts come from `input_batch.num_computed_tokens_cpu` (the same buffer the runner builds positions from). Under async scheduling with speculative decoding, the host copy is optimistic, so in that case the corrected device copy `runner.num_computed_tokens` is read instead.
- **After:** A `[0]` only and tokens unchanged; B delta at the target with norm ratio 1.0001 and cos 1.0; C exact at 10, 74 and 138 (`run_results.tsv`, `fx_suite_*`, `fx_chunked_q25`).

### 2. Pre-hooks never run on Qwen3-Next / Qwen3.5 / Qwen3.6, and flood the log (critical for this project)
- **Cause:** the pre-hook read `args[1]`. `Qwen3NextModel` (and the Qwen3.5/3.6 classes built on it) call `layer(positions=..., hidden_states=..., residual=...)` by keyword, so `args == ()`. Every pre-hook call raised `IndexError`, which was caught and logged *with a traceback*, on every layer of every forward pass once hooks were installed (that is, after any capture or steering request), and the user's function never ran.
- **Measured:** 960 hook-error tracebacks in one short upstream suite on Qwen3.5; 14,232 in one benchmark run. The pre-hook call count was `None` and zeroing the input had no effect.
- **Fix:**
  - Pre-hooks are registered with `with_kwargs=True`, and inputs are located by the layer's `forward` signature (also fixes layers whose positional order differs, e.g. GPT-OSS, LoopCoder).
  - **Deliberate behaviour change:** on fused-residual layers (detected by vLLM's `if residual is None:` pattern) after layer 0, a pre-hook sees `hidden_states + residual` (the full stream entering the layer, equal to the previous layer's post-hook view) instead of only the previous layer's MLP output. A returned tensor replaces that stream. Layer 0 and non-fused layers (Command-R, PhiMoE, EXAONE-4, which pass a stale residual) keep upstream behaviour.
- **After:** calls counted, zeroing changes the capture, pre-hook input at L equals post-hook output at L−1 bit-for-bit, 0 hook errors.

### 3. Hook hot path is very slow (performance)
- **Cause:** on every forward pass, each decoder layer's hook looped over every request, string-matched its id against every registered steering and hook key (O(layers × requests × keys)), did `.item()` device syncs per request per layer, and captured with a blocking `.cpu()` per request per layer. The plugin also forced `enforce_eager=True` on every engine, even ones never using lens features (this is what cost the executor 2 h overnight).
- **Fix** (approach taken from ceselder's vllm-metamodels, re-implemented on 1.3.0 so all upstream features stay):
  - Each request's wants are resolved once (`_ReqPlan`, indexed prefix lookup), cached until the registered data changes.
  - One plan per forward pass from host-side runner buffers (`_StepPlan`, no device syncs); layers with nothing to do return after one lookup.
  - Steering for a layer is one `index_add_` where per-token semantics allow it.
  - Capture is one gathered asynchronous pinned device-to-host copy per layer.
  - The offline path sends steering and hooks in one RPC per `generate()`; batch retrieval matches request ids in linear time.
- **CUDA graphs:** torch.compile is turned off (its graph would bypass the hooks) and `cudagraph_mode=FULL_DECODE_ONLY` stays on. The runner's batch dispatch (`_determine_batch_execution_and_padding`, which has a `force_eager` flag in vLLM 0.30) is told to run eagerly exactly on passes whose plan has hook work, so every hook still fires and passes without hook work replay graphs. `VLLM_LENS_CUDA_GRAPHS=0` restores forced eager mode.

### 4. The plugin mutated the caller's `SamplingParams`
- **Cause:** it popped `apply_steering_vectors` / `apply_hooks` out of the user's own `extra_args`, so a `SamplingParams` reused for a second `generate()` silently ran without steering. Found because the first benchmark measured unsteered generation on its second repetition.
- **Fix:** the offline and async paths work on shallow copies with private `extra_args`. Regression test: `test_reused_sampling_params_keep_steering`.

## Things I got wrong along the way (and how they were caught)

1. **First CUDA-graph version silently skipped hooks on decode steps.** The dispatch wrapper called `_current_plan()`, which required a forward context; vLLM sets that context only after the graph/eager decision, so the wrapper never forced eager, and decode passes with capture work replayed graphs with no hooks. Only the prompt rows were captured. Caught by known-answer check B (`captured_rows` 12 instead of 19); independently flagged critical by two reviewers. Fixed by planning from host buffers only, with no forward-context requirement.
2. **Issues the independent review confirmed in my new code, all fixed:**
   - A malformed `output_residual_stream` could raise inside vLLM's batch dispatch and crash the engine (now isolated per request; planning errors fall back to eager).
   - Pinned capture blocks stayed page-locked after retrieval (now copied out once per block as soon as the transfer lands).
   - Cached plans of finished requests kept cleared steering tensors alive on the GPU (cache dropped on every data change).
   - `prefetch_parameters` called before any other lens RPC was discarded.
   - Captures stayed on the device on non-CUDA accelerators.
   - Pre-hooks after layer 0 corrupted the stream on layers that ignore their residual argument.
   - A pre-hook's saved input view was overwritten.
   - The positional fallback misread some layer signatures.
3. **Invalid first benchmark** (bug 4 above). The benchmark was fixed to build fresh `SamplingParams` each repetition, and rerun.
4. **Looked like a capture regression, wasn't:** one benchmark showed capture slower than upstream. Re-measured back to back on an otherwise idle GPU, capture is faster than upstream (table below); the earlier number was noise from the shared machine.
5. **Operational:**
   - One baseline engine crashed on vLLM's memory-profiling assertion when the executor's job released GPU memory mid-startup. Later runs pass `kv_cache_memory_bytes`, which skips that check.
   - Two of my own engine processes were orphaned when I stopped a run and had to be cleaned up.
   - None of this affected the executor's jobs.

## Benchmarks (batch 256, 64 new tokens, best of 3; noisy machine, ±20–50 %)

Qwen2.5-0.5B (`fx_bench_q25` / `up_bench_q25`):

| case | upstream | fixed |
|---|---|---|
| plain generate | 1.47 s | 0.53 s |
| plain, hooks installed | 1.70 s | 0.59 s |
| one 2-D steering vector per request | 22.3 s | 1.74 s |
| one prompt-position vector per request (activation-oracle style) | 21.9 s | 0.73 s |

Qwen3.5-0.8B (`fx_bench_q35` / `up_bench_q35`):

| case | upstream | fixed |
|---|---|---|
| plain generate | 5.80 s | 1.25 s |
| plain, hooks installed | 8.62 s (14,232 tracebacks) | 1.02 s |
| one 2-D steering vector per request | 27.9 s | 4.64 s |
| one prompt-position vector per request | 25.2 s | 1.52 s |

Prefill-only capture, 3 layers, 256 prompts, run back to back with the GPU otherwise idle (`prof/q35_*`): upstream 5.7–8.4 s per call, fixed 2.1–3.7 s (graphs) and 2.5–5.9 s (eager). In all versions most of this is the retrieval RPC (1–5 s).

## Verification status

- **CPU tests: 208 passed, 1 skipped** (upstream's unit tests, updated plugin-config tests, new randomized equivalence tests showing the indexed lookups reproduce upstream's linear scans exactly).
- **GPU known-answer checks (`run_results.tsv`):** positions A/B/C, pre-hooks and HF parity all pass on both models (min per-token cos vs HF bf16: Qwen2.5 0.99998 / 0.99980 / 0.99919, Qwen3.5 0.99996 / 0.99906 / 0.99928 at layers 0 / 12 / 22), with 0 hook errors.
- **GPU pytest (stopped at the user's request): 30 passed, 5 skipped (multi-GPU), 0 failed.**
  - That covers all 11 new tests in `vllm_lens/tests/test_step_plan.py`, including bit-identical tokens and activations between a CUDA-graph engine and an eager engine.
  - It also covers upstream's async/offline steering, offline capture vs HF, and part of the remaining suite.
  - **Not reached:** upstream's chunked-prefill capture, ordered capture, Q/K exactness / chunked Q/K, layer discovery and cross-API modules.

## Open items

1. Finish the GPU suite (the modules not reached above). Upstream's fixtures keep vLLM's ~10-minute startup JIT warmup, so the first engine is slow.
2. One run on Qwen3.6-27B itself (`scripts/lens_diagnostics.py suite --model <path> --language-model-only --kv-cache-gb 20`). Everything so far is on 0.5B and 0.8B models.
3. Capture retrieval inside the engine still costs 1–2 s per batch on Qwen3.5 even though the same CPU work takes ~35 ms in isolation. `lens_stats` now has `retrieval_flush_s` / `retrieval_build_s` / `retrieval_pickle_s` to locate it; not yet measured.
4. Known approximation: with `norm_match=True` the batched norm can differ from upstream's per-token norm in the last fp32 bit (documented in `_steer_layer`).
5. Not ported from metamodels: last-k capture, in-engine readout, early exit, LoRA merge, prefix caching with steering. These would make prompt-only capture runs (002/003) faster.

## vllm-metamodels (ceselder), for comparison

Read, not run here. It is a fork of vllm-lens 1.1.0.
- **Same fixes:** it fixes the hot-path speed and the position bug the same way.
- **Missing upstream features:** it has none of upstream's later features (generic pre/post/persistent hooks, Q/K capture, registry layer discovery).
- **Graph mode limits:** CUDA graphs are opt-in and prompt-only (2-D steering is refused, generated tokens are not captured).
- **Versions:** tested on vLLM 0.16–0.28. The executor's parity check of it (prompt capture on Qwen3.6-27B, vLLM 0.28) passed.
