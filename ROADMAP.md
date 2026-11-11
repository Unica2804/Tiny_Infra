# Tiny_Infra — Roadmap

Status as of Nov 2026. Based on the repo audit on the day this was written (see below).

## How to read this

Each phase has a goal, concrete tasks, and acceptance criteria. Phases land behind
flags so the engine never regresses. Order is dependency-driven, not calendar-order.

**Standing rules**
- Every phase that touches the math lands with a numerical-equivalence test
  (greedy-token agreement or logit parity vs the FP16 baseline on a fixed prompt suite).
- One kernel language only: **Triton**. No `.so`, no build step, no ABI fragility.
- One dependency source of truth: `pyproject.toml` (+ `uv.lock`). No `requirements.txt`.
- **Plug-and-play checkpoints only**: the engine loads native HF `.safetensors`
  (hub id or local dir) with HF layer naming as-is. No custom layouts, no
  extraction/translation step, no renamed weight files.

---

## Repo audit (the short version)

What exists: Qwen2.5-0.5B FP16 inference (custom modules), paged KV cache, continuous
batching, streaming SSE, FastAPI server, Textual TUI, Triton fused MLP.

What's broken: prebuilt CUDA extensions unloadable in the current venv (CUDA 12 vs
13 mismatch) → the engine does not boot. All 5 test files broken (stale imports).
`gather(task2)` drops a request; TUI sends `top_p` (API expects `p_value`); streaming
disconnect never cancels engine-side generation; one decode error kills the whole
batch; dependency files drift (requirements.txt vs uv.lock); committed `*.so` in git;
945 MB unused AWQ checkpoint (Qwen3.5-0.8B, incompatible architecture) sitting in
`weights/`; engine can't load a native checkpoint — it requires a custom-flat
safetensors produced by `utils/extractor.py`.

---

## Phase 0 — Snapshot the current state
*(½ day)*

- [ ] Commit or stash the uncommitted changes (`pyproject.toml`, `uv.lock`, `tui_client.py`).
- [ ] Tag `pre-roadmap` so this state is always recoverable.
- [ ] `git rm --cached engine/ops/*.so` and add `*.so` to `.gitignore`.

**Done when:** working tree is clean on a tagged commit.

## Phase 1 — Cleanup & installable package
*(3–5 days)*

Restructure flat layout into a package:

```
src/tinyinfra/
  engine/    (scheduler, cache_manager, loader, sampler)
  models/    (config, qwen2)
  kernels/   (fused_mlp, + phase 2 additions)
  api/       (schema, router, server)
  client/    (chat client, Textual TUI)
tests/       (fixed suite + parity harness)
scripts/
```

- [ ] `pyproject.toml` becomes the only dep file; `uv.lock` regenerated; delete `requirements.txt`.
- [ ] Console entry points: `tinyinfra-serve`, `tinyinfra-tui`; `uv run` works.

**Native-checkpoint loading (plug-and-play).** The HF key naming is used by the
modules as-is — the translation lived entirely in the weight file, so native loading
is loader work, not model work:
- [ ] `load_model(model_ref, ...)`: `model_ref` = HF hub id or local snapshot dir;
      merge-load all `.safetensors` shards with `strict=True`; clear error on
      missing/unexpected keys (lists them).
- [ ] `QwenConfig` built from the checkpoint's `config.json` (vocab_size, hidden_size,
      num_layers, heads, kv-heads, intermediate_size, rope_theta, rms_norm_eps,
      tie_word_embeddings, max_position_embeddings) — no hardcoded 0.5B values;
      unknown `model_type` → explicit error listing supported archs.
- [ ] `tie_word_embeddings`: alias `lm_head.weight` → `model.embed_tokens.weight`
      when the checkpoint omits it (Qwen2.5-0.5B ties).
- [ ] KV `max_seq_len` is a **serving-window flag** (default 1024/4096), never
      `max_position_embeddings` (Qwen2.5 reports 131,072 — don't allocate for it).
- [ ] EOS from `generation_config.json` (`eos_token_id` may be a list; chat models
      commonly stop on `<|im_end|>` ≠ tokenizer default).
- [ ] Delete `utils/extractor.py`, the custom naming, and the 2.5 GB extracted file;
      `weights/` is optional download cache only. No backwards-compat shim.
- [ ] `--model` flag on serve; tokenizer strictly from the same hub id/dir.
- [ ] Bugfix pass (from audit): `main.py` gather both tasks; `top_p`→`p_value` in TUI;
      SSE disconnect → `stream_gen.aclose()` in `finally`; per-request error isolation in decode;
      `finish_reason` from real stop reason; throughput counters count tokens not sequences;
      drop CUDA sync in `_report_throughput`; delete dead/commented code; logger not `print`.
- [ ] Kill or fix broken tests; add `pytest`/`pytest-asyncio` to dev deps; `[tool.pytest.ini_options]`.
- [ ] CI (GitHub Actions): ruff lint + pytest (CUDA-marked tests skip on CPU runners).
- [ ] README: honest quickstart (install, `tinyinfra-serve --model Qwen/Qwen2.5-0.5B-Instruct`, chat).

**Done when:** fresh clone → `uv sync` → `uv run pytest` green → `tinyinfra-serve --model <hf-id>` boots
from the untouched hub snapshot (no extraction step) → `tinyinfra-tui` holds a multi-turn conversation.

## Phase 2 — Triton kernel port
*(4–7 days)*

Delete the CUDA path entirely; nothing to compile, nothing to pin.

- [ ] RoPE → Triton (in-place, cos/sin tables, GQA over kv-heads).
- [ ] Paged decode attention → Triton (flash-style online softmax; block-table
      indirection; GQA `kv_head = head // (heads/kv_heads)`; fp16 in, fp32 accum).
      Prefill keeps PyTorch SDPA (it's flash-backed) unless Triton prefill wins benchmarks.
- [ ] RMSNorm → Triton (fuse residual add; fp32 variance).
- [ ] Delete `csrc/`, `setup.py`, `engine/ops/`; remove `custom_swiglu` (unused;
      fused-MLP Triton kernel already subsumes it).
- [ ] Parity harness: random-tensor tests, engine-vs-SDPA, engine-vs-HF logits on a fixed
      prompt (greedy token agreement). Run in CI.
- [ ] Microbenchmarks per kernel vs old CUDA path.

**Done when:** no `.so` anywhere, engine produces identical greedy output vs HF
reference on the parity prompt, tok/s no worse than the old kernels.

## Phase 3 — Fused inference path
*(2–4 days)*

Maximum fusion, measurable, flag-guarded:

- [ ] Add + RMSNorm fused Triton kernel (kill two elementwise passes).
- [ ] Attention-out + residual + post-attention-norm fused.
- [ ] Gate/up GEMM + SwiGLU (exists) and down-proj + residual (verify, then keep).
- [ ] Optionally `torch.compile` on the remaining glue (norms, residuals) behind a flag.
- [ ] LM head GEMM left unfused (beneficial to fuse only with sampling argmax — later).
- [ ] Benchmark table (prefill ms, decode tok/s, VRAM) per fusion flag.

**Done when:** each fusion individually togglable, parity tests pass with it on,
README/benchmark table shows the wins.

## Phase 4 — Quantization (load external checkpoints only)
*(2–4 weeks, staged)*

No quantizer in-engine. Builds on the Phase-1 native loader: a `QuantConfig` parsed
from the checkpoint's `quantization_config.json` (`awq` / `gptq` / llm-compressor
`smoothquant`-style).
Weights stay packed/int8 on GPU; dequant happens in-kernel; activations stay FP16
(except W8A8). GPTQ support falls out of the AWQ loader (same `qweight/qzeros/scales`
family).

Stage A — **W8A16** (per-output-channel int8 weights, fp16 activations)
- [ ] Triton dequant+GEMM (int8 load → scale → fp16 dot); int8 weights resident.
- [ ] Loader/config: `quant_method: int8`, static per-channel scales from checkpoint.

Stage B — **W4A16 AWQ** (group-128, symmetric or asymmetric)
- [ ] Triton AWQ-GEMM: unpack int4 from packed int32 (`qweight [K, N/8]`), apply
      `scales`/`qzeros` (sym → no zero-point math), fp16 dot.
- [ ] Validate on a Qwen2.5-0.5B AWQ checkpoint (pull from Hub; **not** the on-disk
      Qwen3.5 file). Acceptance: ≥98% greedy-token agreement vs FP16 on fixed suite;
      VRAM table shows ~55–65% of FP16.

Stage C — **W8A8** (llm-compressor / SmoothQuant-format checkpoints)
- [ ] Parse activation scheme from checkpoint config (static per-tensor input
      scales, or per-token dynamic — support both).
- [ ] Triton `int8×int8 → int32` tensor-core GEMM; requantize per-token activations
      (or read static scales); per-channel static weight dequant at the end.

Stage D — **W4A8** — stretch only; documented, unscheduled.

_Note for scope: Qwen2.5-0.5B quantizes poorly (small model). Correctness is audited
at 0.5B; the real showcase is 7B-class in Phase 5._

**Done when:** AWQ W4A16 + W8A8 checkpoints load and serve; parity + tok/s + VRAM
measured and documented; all runtime tests green on every supported quant format.

## Phase 5 — Streaming weights (engine feature, model-agnostic)
*(2–3 weeks)*

Generic capability: for any supported model, stream weights layer-by-layer from an
mmap'd file to VRAM, overlapping H2D copies with compute. Not model-specific.

- [ ] safetensors loaded with `safe_open` (lazy, mmap-backed; NVMe-friendly).
- [ ] Pinned ping-pong staging buffers; dedicated CUDA copy stream; prefetch depth
      (default 2 layers) configurable.
- [ ] Forward pass streams each transformer layer's weights into the buffer for the
      current layer; evict after use; KV cache/activations/logits stay on GPU.
- [ ] batching amortizes per-pass weight traffic (stream once per step regardless of
      batch size) — streaming helps throughput per request under load.
- [ ] **lm_head handled explicitly**: for large-vocab models it can be ~20% of per-token
      bytes; stream it like a layer, or keep resident when only lm_head fits; configurable.
- [ ] Embeddings: keep resident (cheap) or stream — configurable.
- [ ] Quant × stream composition: `--stream-weights` + an AWQ W4A16 checkpoint halves
      every byte moved (documented as the headline pairing; 7B W4 ≈ 4.5 GB vs 14 GB FP16).
- [ ] Boot-time VRAM budget check: report weights resident budget, KV cache allocation,
      streaming buffers; refuse (with a clear message) if KV + 2 streaming buffers don't fit.
- [ ] Metrics: bytes streamed/s, prefetch stall time, per-layer wait time exposed in logs.

**Done when:** a model larger than VRAM generates correctly at a measured tok/s
(e.g., Qwen2.5-7B-Instruct AWQ on the 4 GB RTX 2050), and the feature is a config
flag, not a code path fork.

## Phase 6 — Serving hardening (prototype for other tools/users)
*(1–2 weeks)*

- [ ] OpenAI-compatible API: `POST /v1/chat/completions` (SSE `data: [DONE]`),
      `GET /v1/models`; keep native `/generate`; map finish reasons (`stop`/`length`).
- [ ] Backpressure: bounded request queue; 429/503 with `Retry-After` on saturation.
- [ ] Auth: optional bearer token via env (`TINYINFRA_API_KEY`); per-key rate limit
      (token bucket); CORS configurable.
- [ ] `GET /healthz`, `GET /readyz`; graceful drain on SIGTERM (in-flight finish,
      new requests refused).
- [ ] Client-disconnect → engine cancels generation and frees the slot (end-to-end).
- [ ] Prometheus metrics: active sequences, queue depth, TTFT histogram, windowed tok/s;
      `X-Request-ID` on responses.
- [ ] Dockerfile (optional) + documented `uvicorn` deployment.

**Done when:** an OpenAI SDK client (`openai` Python package) works against the
server unchanged; under overload the server refuses cleanly instead of OOMing.

## Phase 7 — Qwen3.5 hybrid architecture (late milestone)
*(quarter-scale)*

The on-disk `Qwen3.5-0.8B-W4A16-AutoRound-AWQ` finally becomes usable.

- [ ] Linear-attention (gated delta-net) layer: fused Triton kernel (reuse the
      Phase-2 infrastructure; decay state per layer).
- [ ] Delta-net KV/state cache: fixed-size per-sequence state; extend `KVCacheManager`
      (not block-paged — state is constant size; block tables don't apply).
- [ ] Interleaved full-attention every N layers; partial rotary (0.25 factor);
      `q_norm`/`k_norm`; tied embeddings; vocab 248,320.
- [ ] Text-only first (skip visual tower; reject image prompts with a clear error).
- [ ] mRoPE sections (per `rope_parameters.mrope_section`) if multi-modal later.
- [ ] W4A16 path reused as-is (same packing format; verify shapes).

**Done when:** text-only greedy generation on the AWQ checkpoint matches HF
reference, and it serves through the normal API/TUI.

## Phase 8 — Benchmark & eval harness (continuous)

- [ ] `scripts/bench.py`: fixed prompt suite; TTFT, tok/s, VRAM, total latency,
      greedy-agreement vs HF baseline; results table committed to README.
- [ ] Baselines recorded at FP16, W8A16, W4A16, (W8A8), streamed modes.
- [ ] Perplexity spot-check on a small corpus for quant regressions (0.5B as warning bell).
- [ ] CI smoke test: generate 32 tokens on CPU (numerical checks only).

---

## Dependencies & order

```
P0 → P1 → P2 → P3 ─────────────────────┐
          └→ P4A → P4B ─→ P5 (streaming) → P6 → P7
                     └→ P4C (W8A8) → P8 (harness, ongoing)
```

Phase 5 depends on Phase 4 Stage B being correct (streaming FP16 is not viable).
Phase 6 is independent of 4–5 and can overlap. P7 needs P2's Triton infrastructure.

## Explicitly out of scope (for now)

W4A8 (stretch only), prefix caching, speculative decoding, multi-GPU, distributed
serving, LoRA/adapters, embeddings endpoint, in-engine quantization (calibrate).

## Risk register

| Risk | Mitigation |
|---|---|
| Triton paged-attention correctness | parity fuzz vs SDPA (shapes × block tables × GQA); block-table property tests |
| Quant accuracy loss on 0.5B | validate on 0.5B; real demo on 7B in P5; agreement metric gates CI |
| AWQ/GPTQ key-layout variance across checkpoints | tolerant loader on top of the Phase-1 native loader; one validated reference checkpoint per format; clear errors on unknown layouts |
| HF `config.json` variance across Qwen archs/sizes | tolerant config parsing with validated reference per arch (Qwen2 dense family first); unknown `model_type` → explicit error, never silent defaults |
| llm-compressor config variance for W8A8 | same strategy — read `quantization_config.json`, tolerate static per-tensor and dynamic per-token activation schemes; validated reference checkpoint required |
| Streaming slower than hoped (PCIe 3.0 x16 ≈ 16 GB/s) | W4 streaming (≈4.5 GB) as headline; benchmark prefetch depth; page-cache warm reads; document floor formula `bytes/PCIe_bw` |
| VRAM pressure (weights + KV cache) | boot-time budget check; KV cache after weights; streaming buffers bounded at 2 layers |
| P7 scope creep | text-only, flag-gated; P7 only after P6 ships |
| Dependency drift again | `pyproject.toml` is the single source; CI reinstalls from scratch |

## Rough effort

P0 half day · P1 4–7 days · P2 4–7 days · P3 2–4 days · P4 2–4 weeks · P5 2–3 weeks ·
P6 1–2 weeks · P7 quarter-scale · P8 ongoing. Core (P0–P6): ~2–3 months part-time.
