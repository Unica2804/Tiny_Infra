import torch

def warmup_engine(model, config, kv_cache, cos_cache, sin_cache, *, device="cuda", prefill_tokens=32, decode_steps=2) -> None:
    "Triggers JIT+ autotune for every kernel used by prefill and decode"
    if device != "cuda":
        return  # Triton kernels are CUDA-only; nothing to compile on CPU
    bi = torch.tensor([0], device=device, dtype=torch.int32)

    ids = torch.randint(0, config.vocab_size, (1,prefill_tokens), device=device, dtype=torch.int32)

    with torch.no_grad():
        model(ids,bi,kv_cache,cos_cache,sin_cache)
    kv_cache.seq_len[bi] += prefill_tokens

    for _ in range(decode_steps):
        tok = torch.randint(0, config.vocab_size, (1,1), device=device, dtype=torch.int32)

        with torch.no_grad():
            model(tok,bi,kv_cache,cos_cache,sin_cache)
        kv_cache.seq_len[bi] += 1
    torch.cuda.synchronize()
    kv_cache.free_batch_slot(0)
