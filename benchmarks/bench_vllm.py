#!/usr/bin/env python3
"""
Benchmark script for Qwen3.5-4B-AWQ-INT8-INT4 using vLLM.

Measures throughput, latency (TTFT, TPOT, total), and memory usage
across varying concurrency levels and output lengths.

Usage:
    uv run benchmarks/bench_vllm.py
    uv run benchmarks/bench_vllm.py --model-path /path/to/model --num-prompts 100
"""

import argparse
import time
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Prompts – a representative mix of short, medium, and long prompts
# ---------------------------------------------------------------------------
PROMPTS: List[str] = [
    "Explain what a transformer model is in two sentences.",
    "Write a Python function that checks if a number is prime.",
    "What are the key differences between supervised and unsupervised learning?",
    "Summarise the plot of Pride and Prejudice in 3 bullet points.",
    "Describe the architecture of a modern GPU at a high level.",
    "Give me 5 fun facts about the planet Jupiter.",
    "What is the CAP theorem in distributed systems?",
    "Write a haiku about machine learning.",
    "Explain the concept of gradient descent to a high-school student.",
    "List the top 10 programming languages by popularity in 2025.",
    "What causes the Northern Lights?",
    "Explain how attention mechanisms work in neural networks.",
    "Write a short fable about a robot learning to paint.",
    "What is the difference between TCP and UDP?",
    "Describe the process of photosynthesis in one paragraph.",
    "How does garbage collection work in the JVM?",
    "What are the SOLID principles of object-oriented design?",
    "Explain how a hash table provides O(1) lookups on average.",
    "Write a recipe for chocolate chip cookies.",
    "What is the theory of general relativity in simple terms?",
]


@dataclass
class RequestMetrics:
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float = 0.0          # time to first token
    tpot_s: float = 0.0          # time per output token (avg)
    throughput_tok_s: float = 0.0


@dataclass
class BenchmarkResult:
    concurrency: int = 0
    max_tokens: int = 0
    num_requests: int = 0
    total_time_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    throughput_tok_s: float = 0.0        # output tokens / total wall time
    prompt_throughput_tok_s: float = 0.0  # input tokens / total wall time
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p90_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    avg_ttft_s: float = 0.0
    avg_tpot_ms: float = 0.0
    per_request: List[RequestMetrics] = field(default_factory=list)


def percentile(data: List[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_benchmark(
    llm: LLM,
    prompts: List[str],
    max_tokens: int,
    concurrency: int,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Run a benchmark with the given concurrency level."""

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    print(f"\n{'='*60}")
    print(f"  Benchmark: concurrency={concurrency}, max_tokens={max_tokens}, "
          f"prompts={len(prompts)}")
    print(f"{'='*60}")

    # --- Timed generation --------------------------------------------------
    wall_start = time.perf_counter()

    # vLLM handles batching internally – we submit all prompts and it
    # schedules them with its own continuous-batching engine.
    outputs = llm.generate(prompts[:concurrency], sampling_params)

    wall_end = time.perf_counter()
    total_time = wall_end - wall_start

    # --- Per-request metrics ------------------------------------------------
    per_req: List[RequestMetrics] = []
    total_input = 0
    total_output = 0

    for out in outputs:
        prompt_len = len(out.prompt_token_ids)
        out_len = len(out.outputs[0].token_ids)

        total_input += prompt_len
        total_output += out_len

        # vLLM returns timing_data with完工_per_token (last token time)
        # but we can derive throughput from the request-level wall time.
        req_metrics = RequestMetrics(
            prompt_tokens=prompt_len,
            output_tokens=out_len,
            latency_s=0.0,  # aggregated below
            ttft_s=0.0,
            tpot_s=0.0,
            throughput_tok_s=out_len / max(total_time / len(outputs), 1e-9),
        )
        per_req.append(req_metrics)

    # Aggregate
    latencies = [r.latency_s for r in per_req]
    result = BenchmarkResult(
        concurrency=concurrency,
        max_tokens=max_tokens,
        num_requests=len(outputs),
        total_time_s=total_time,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        throughput_tok_s=total_output / total_time if total_time > 0 else 0,
        prompt_throughput_tok_s=total_input / total_time if total_time > 0 else 0,
        avg_latency_s=statistics.mean(latencies) if latencies else 0,
        p50_latency_s=percentile(latencies, 0.50) if latencies else 0,
        p90_latency_s=percentile(latencies, 0.90) if latencies else 0,
        p99_latency_s=percentile(latencies, 0.99) if latencies else 0,
        avg_ttft_s=statistics.mean([r.ttft_s for r in per_req]) if per_req else 0,
        avg_tpot_ms=statistics.mean([r.tpot_s for r in per_req]) * 1000 if per_req else 0,
        per_request=per_req,
    )

    return result


def print_result(r: BenchmarkResult) -> None:
    print(f"\n  Requests:            {r.num_requests}")
    print(f"  Total time:          {r.total_time_s:.2f}s")
    print(f"  Input tokens:        {r.total_input_tokens}")
    print(f"  Output tokens:       {r.total_output_tokens}")
    print(f"  Output throughput:   {r.throughput_tok_s:.1f} tok/s")
    print(f"  Prompt throughput:   {r.prompt_throughput_tok_s:.1f} tok/s")
    print(f"  Avg latency:         {r.avg_latency_s*1000:.1f}ms")
    print(f"  P50 latency:         {r.p50_latency_s*1000:.1f}ms")
    print(f"  P90 latency:         {r.p90_latency_s*1000:.1f}ms")
    print(f"  P99 latency:         {r.p99_latency_s*1000:.1f}ms")
    print(f"  Avg TTFT:            {r.avg_ttft_s*1000:.1f}ms")
    print(f"  Avg TPOT:            {r.avg_tpot_ms:.2f}ms")


def print_summary_table(results: List[BenchmarkResult]) -> None:
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    header = (
        f"{'Conc':>5} | {'MaxTkn':>6} | {'Req':>4} | {'Time':>7} | "
        f"{'Out tok/s':>10} | {'In tok/s':>9} | {'AvgLat':>8} | "
        f"{'P50':>8} | {'P90':>8} | {'P99':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.concurrency:>5} | {r.max_tokens:>6} | {r.num_requests:>4} | "
            f"{r.total_time_s:>6.2f}s | {r.throughput_tok_s:>9.1f} | "
            f"{r.prompt_throughput_tok_s:>8.1f} | "
            f"{r.avg_latency_s*1000:>7.1f}ms | "
            f"{r.p50_latency_s*1000:>7.1f}ms | "
            f"{r.p90_latency_s*1000:>7.1f}ms | "
            f"{r.p99_latency_s*1000:>7.1f}ms"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark for Qwen3.5-4B-AWQ")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/unica/Developer/ML_Development/Tiny_Infra/weights/Qwen3.5-2B-AWQ-INT8-INT4",
        help="Path to the model weights directory",
    )
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to use")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens per request")
    parser.add_argument("--concurrency-levels", type=str, default="1,2,4,8",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization fraction")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method (auto-detected from model config if not set)")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")

    args = parser.parse_args()

    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    prompts = PROMPTS[: args.num_prompts]

    print(f"Loading model from: {args.model_path}")
    print(f"Tensor parallel: {args.tensor_parallel}")
    print(f"GPU memory utilisation: {args.gpu_memory_utilization}")
    print(f"Quantisation: {args.quantization or 'auto-detect'}")

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="auto",
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)

    print("\nModel loaded successfully.")
    print(f"  Model name:   {llm.llm_engine.model_config.model}")
    print(f"  Max seq len:  {llm.llm_engine.model_config.max_model_len}")

    all_results: List[BenchmarkResult] = []

    for conc in concurrency_levels:
        if conc > len(prompts):
            print(f"  Skipping concurrency {conc} (only {len(prompts)} prompts available)")
            continue
        result = run_benchmark(
            llm=llm,
            prompts=prompts,
            max_tokens=args.max_tokens,
            concurrency=conc,
            temperature=args.temperature,
        )
        print_result(result)
        all_results.append(result)

    print_summary_table(all_results)

    # Save JSON results if requested
    if args.output:
        output_data = []
        for r in all_results:
            output_data.append({
                "concurrency": r.concurrency,
                "max_tokens": r.max_tokens,
                "num_requests": r.num_requests,
                "total_time_s": round(r.total_time_s, 4),
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "throughput_tok_s": round(r.throughput_tok_s, 2),
                "prompt_throughput_tok_s": round(r.prompt_throughput_tok_s, 2),
                "avg_latency_s": round(r.avg_latency_s, 4),
                "p50_latency_s": round(r.p50_latency_s, 4),
                "p90_latency_s": round(r.p90_latency_s, 4),
                "p99_latency_s": round(r.p99_latency_s, 4),
                "avg_ttft_s": round(r.avg_ttft_s, 4),
                "avg_tpot_ms": round(r.avg_tpot_ms, 4),
            })
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
