# Building a Custom Inference Engine: A Journey from Scratch 

**Abstract:**
This document details the architecture, design decisions, and engineering challenges of building a custom, hardware-accelerated Large Language Model (LLM) inference engine from the ground up. Targeting the Qwen2.5-0.5B-Instruct architecture, the project eschewed high-level abstractions in favor of bare-metal PyTorch and custom CUDA kernels. The system features a Paged KV Cache, an asynchronous Continuous Batching Scheduler, and fused GPU kernels. This post chronicles the development lifecycle, highlighting the subtle mathematical and structural bugs encountered during implementation and the methodologies used to resolve them.

---

## Architecture & Data Flow Breakdown

```
+-----------------------------------------------------------------------------------+
|                                  USER / CLIENT                                    |
+-----------------------------------------------------------------------------------+
       |                                                                   ^
       | 1. generate_stream(prompt_tokens, ...)                            | 6. Yield Tokens
       v                                                                   |
+-----------------------------------------------------------------------------------+
|                                CONTINUOUS BATCHER                                 |
|                                                                                   |
|  Requests Queue: [ GenerationRequest_1, GenerationRequest_2, ... ]               |
|  Active Slots:   [ Slot_0 (Req 1) | Slot_1 (Req 2) | Slot_2 (None) | Slot_3 (None) ] |
+-----------------------------------------------------------------------------------+
       |                                                                   ^
       | 2. Extract Active Requests                                        | 5. Store Tokens &
       |    & Batch Tensors                                                |    Check EOS/Max
       v                                                                   |
+-----------------------------------------------------------------------------------+
|                         PYTORCH MODEL & CUDA KERNELS                              |
|                                                                                   |
|  - Qwen2ForCausalLM Forward Pass                                                  |
|  - RoPE Embeddings & Paged Attention Kernel                                       |
|  - Vectorized Temperature / Top-K / Top-P Sampling                                |
+-----------------------------------------------------------------------------------+
       |                                                                   ^
       | 3. Read/Write KV Blocks                                           | 4. Update Slot
       v                                                                   |    Sequence Length
+-----------------------------------------------------------------------------------+
|                                 KV-CACHE MANAGER                                  |
|                                                                                   |
|  - Physical Block Allocator (Block Size = 16)                                     |
|  - Slot-to-Block Mapping Tables                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 1. Architectural Foundation: The Explicit Layout

Modern LLM frameworks (like HuggingFace `transformers`) are built for general-purpose flexibility, often at the cost of highly localized optimizations. To maximize efficiency, we defined an **explicit custom engine layout**. 

Instead of relying on dynamic configuration graphs, we mapped the raw `.safetensors` weights directly to our engine's architecture.

**Design Decision:** We wrote a custom extractor (`extractor.py`) to pull weights from the HuggingFace Hub, bypassing the standard `AutoModelForCausalLM` loading pipeline for inference. The weights were explicitly mapped to custom keys (e.g., `blocks.0.attn.q_proj.weight`) and saved as a flat binary file. This allowed `loader.py` to inject weights directly into our simplified, overhead-free PyTorch module (`qwen_25.py`).

## 2. Memory & Scheduling: Continuous Batching & Paged Attention

Traditional batching forces the engine to wait for the longest sequence in a batch to finish before accepting new requests, wasting immense compute. To solve this, we implemented **Continuous Batching** alongside a **Paged KV Cache**.

* **Paged KV Cache (`cache_manager.py`):** Instead of allocating contiguous memory tensors for maximum context lengths (which fragments VRAM), we implemented a virtual memory system. The KV cache allocates memory in non-contiguous "blocks" (e.g., 16 tokens per block). A routing table (`block_tables`) maps logical token positions to physical blocks.
* **Continuous Batching Scheduler (`scheduler.py`):** An asynchronous event loop that dynamically swaps completed requests with new ones from the queue at a per-token granularity. It splits execution into two distinct phases:
    1.  **Prefill Phase:** Computes the initial prompt comprehensively.
    2.  **Decode Phase:** Autoregressively generates token-by-token, batching multiple concurrent requests into a unified matrix multiplication.

## 3. Hardware Acceleration: Custom CUDA Kernels

To bypass the overhead of standard PyTorch operations during the decode phase, we wrote custom C++/CUDA extensions:
* **Fused RoPE (`fused_rope.cu`):** Performs Rotary Positional Embeddings entirely in-place, drastically reducing VRAM read/write bandwidth.
* **Paged Attention (`attention.cu`):** A custom scaled dot-product attention kernel that understands our virtual block tables, allowing it to attend to scattered memory blocks seamlessly without gathering them into a contiguous tensor first.
* **Fused SwiGLU (`fused_swiglu.cu`):** Combines the Gated Linear Unit and SiLU activation functions into a single kernel launch.

---

## What Sets This Engine Apart (Comparison)

Unlike standard inference frameworks or naive wrappers, this engine explicitly handles memory management and execution scheduling at the raw kernel interface level:

| Feature / Metric | Naive PyTorch / Transformers | Enterprise (vLLM / TGI) | Our Custom Engine |
| :--- | :--- | :--- | :--- |
| **KV Cache Allocation** | Contiguous padding to `max_seq_len` | Paged Attention (Block Tables) | **Paged KV Allocation (Block Size = 16)** |
| **Batching Strategy** | Static batching (padded strings) | Continuous Batching / Iteration-level | **Continuous Batching with Prefill/Decode Split** |
| **Weight Extraction** | Automatic AutoModel loading | Custom C++ / Rust engines | **Explicit Safetensors Mapper & Loader** |
| **Sampling Execution** | Python loops / HF Generate wrappers | C++ CUDA sampling kernels | **Fully Vectorized PyTorch Tensor Pipeline** |
| **Streaming Mechanism** | Blocking generators / TextStreamer | gRPC / HTTP Server Push | **Native `asyncio.Queue` Sentinel Streams** |
| **Dependency Footprint** | Massive (Transformers, Accelerate) | High (C++ CUDA extensions) | **Minimal (PyTorch + HuggingFace Tokenizer)** |

---

## 4. The Crucible: Diagnosing and Fixing Silent Failures

Building an engine from scratch is an exercise in precision. While compilation errors are easy to fix, LLMs suffer from "silent failures"—where the math executes, but the outputs collapse into gibberish. Here are the major architectural bugs we encountered and how we solved them.

### 4.1 The Sequence Length Desync
**The Problem:** During the forward pass, the cache manager was updating the global `seq_len` tracker dynamically inside the layer loop. By the time Layer 1 executed, it read a sequence length that Layer 0 had already incremented, causing it to write keys and values to the wrong offset.
**The Solution:** We decoupled state mutation from the forward pass. The cache manager was restricted to *reading* `seq_len` during generation. We shifted the `seq_len` increment logic to the very end of the `ContinuousBatcher` step, ensuring a stable snapshot across all 24 transformer blocks.

### 4.2 The "Off-By-One" Attention Blindness
**The Problem:** After stabilizing the `seq_len` update to occur *after* the model forward pass, the Paged Attention kernel during the decode step was starved of its current token. It received `context_lens` tracking up to token $T-1$, completely ignoring the newly inserted token $T$.
**The Solution:** We modified the decode execution branch to dynamically inject the current `q_length` into the attention context (`active_context_lens = context_lens[batch_indices] + q_length`), ensuring the model could attend to its own current state.

### 4.3 The RoPE Base Frequency & Pointer Mismatch
**The Problem:** The engine compiled perfectly, but outputted complete gibberish. The attention mechanism was "positionally blind."
**The Solution:** Two distinct errors were found in the RoPE cache initialization:
1.  **Base Frequency:** We initialized RoPE with `rope_theta = 10000.0` (standard for Llama models). However, Qwen2.5 requires a high-frequency decay rate of `1000000.0`. 
2.  **Pointer Stride Mismatch:** The Python code concatenated the `sin` and `cos` frequencies, passing a tensor with a stride of `head_dim` (64) to the CUDA kernel. However, the C++ kernel pointer arithmetic expected a stride of `half_dim` (32). We eliminated the concatenation in Python, ensuring the memory layouts aligned perfectly across the language barrier.

### 4.4 The Missing Attention Biases (The Hallucination Culprit)
**The Problem:** Even with perfect positional IDs and cache alignment, the model hallucinated aggressively. 
**The Solution:** Standard practice in many older LLMs is to use `bias=False` for Q, K, and V projections. However, Qwen2.5 heavily relies on biases for its attention projections. Our initial `extractor.py` only pulled the `.weight` tensors, silently dropping the `.bias` vectors. We updated the extractor, the loader, and the PyTorch module to explicitly extract, map, and apply the QKV biases.

### 4.5 Managing Sequence Termination (EOS)
**The Problem:** The engine successfully generated coherent text but suffered from runaway generation, only stopping when it hit the arbitrary `max_new_tokens` limit, wasting compute.
**The Solution:** We extracted the `eos_token_id` from the HuggingFace tokenizer and integrated an early-stopping monitor into both the Prefill and Decode phases. If the model predicts `<|im_end|>`, the scheduler immediately marks the request as complete, tears down the active slot, and frees the physical memory blocks back to the available pool. Also applied chat template so the prompt is formatted according to model's generation_config.

### 4.6 Asynchronous Consumer Hanging on Stream Exit (Missing Sentinel)
* **Problem**: Successfully completed requests generated text correctly, but user-side `async for` loops hung indefinitely after the final token.
* **Root Cause**: Inline completion code inside `_execute_prefill` and `_execute_decode` freed KV slots and set `completion_event`, but failed to push the `_END_OF_STREAM` sentinel onto `req.token_queue`.
* **Solution**: Collapsed all completion logic into `_finish_request()`. Both successful completions (EOS / Max Tokens) and runtime errors now call `_finish_request()`, ensuring `_END_OF_STREAM` is placed on the queue.

```python
def _finish_request(self, slot_idx: int, reason: str, error: Optional[Exception] = None):
    req = self.active_slots[slot_idx]
    if req is None:
        return
    req.error = error
    req.completion_event.set()
    req.token_queue.put_nowait(_END_OF_STREAM) # Unblocks async consumer
    self.kv_cache.free_batch_slot(slot_idx)
    self.active_slots[slot_idx] = None
```
### 4.7 GPU Stalls during Sampling (O(N) CPU-GPU Synchronizations)
* **Problem**: Initial sampling implementation used Python loops across batch rows, calling `.item()` on individual row tensors for argmax or multinomial steps.
* **Root Cause**: Calling `.item()` on a CUDA tensor forces a synchronous host-device memory transfer, stalling the CUDA stream on every active request during every decode step.
* **Solution**: Vectorized `_sample_next_token()` using batched PyTorch tensor operations across `dim=0` (temperature scaling, top-k masking, cumulative top-p filtering, and batched multinomial sampling), extracting the result with a single `.tolist()` call at the end.

### 4.8. Multi-Byte UTF-8 Streaming Corruption
* **Problem**: Streaming raw token IDs directly to stdout caused occasional garbled characters when multi-byte UTF-8 sequences (such as emojis or non-ASCII characters) were split across token boundaries.
* **Root Cause**: Calling `tokenizer.decode([token_id])` on a fractional byte sequence produces replacement characters (`�`).
* **Solution**: Implemented accumulated token buffering in the consumer: the client accumulates `generated_ids`, decodes the entire sequence using `tokenizer.decode(generated_ids, skip_special_tokens=True)`, and prints only the new string diff (`text[printed_len:]`).

---

## File Map

- **`engine/scheduler.py`**: Continuous batching scheduler, request queuing, prefill/decode dispatching, and vectorized sampling.
- **`engine/cache_manager.py`**: Paged KV Cache block allocation and physical slot management.
- **`engine/loader.py`**: Safetensors parser and state dict translation mapping.
- **`models/qwen_25.py`**: Qwen2.5 neural network topology with RoPE embeddings and Paged Attention hooks.
- **`main.py`**: Entry point executing concurrent streaming tasks.

---

## 5. Conclusion and Next Steps

The completion of this pipeline marks the successful deployment of a high-performance, continuously batching FP16 engine capable of running Qwen2.5-0.5B locally with near-zero overhead. 

**Future Roadmap:**
1.  **Quantization:** Extending the custom kernels to support INT8/INT4 weight-only quantization to further reduce the VRAM footprint and increase memory bandwidth utilization.