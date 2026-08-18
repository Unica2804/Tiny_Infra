#!/usr/bin/env python3
"""
Simple throughput benchmark — load model, run prompts, measure tok/s.

Usage:
    uv run benchmarks/throughput.py
    uv run benchmarks/throughput.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import time
import torch
from vllm import LLM, SamplingParams

PROMPTS = [
    "Explain what a transformer model is in two sentences.",
    "Write a Python function that checks if a number is prime.",
    "What are the key differences between supervised and unsupervised learning?",
    "Summarise the plot of Pride and Prejudice in 3 bullet points.",
    "Describe the architecture of a modern GPU at a high level.",
    "Give me 5 fun facts about the planet Jupiter.",
    "What is the CAP theorem in distributed systems?",
    "Write a haiku about machine learning.",
    "Explain how attention mechanisms work in neural networks.",
    "What causes the Northern Lights?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/unica/Developer/ML_Development/Tiny_Infra/weights/Qwen3.5-0.8B-W4A16-AutoRound-AWQ")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of rounds to repeat all prompts")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Loading model...")

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="auto",
        enforce_eager=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    all_output_tokens = 0
    all_output_text = 0
    latencies = []

    total_t0 = time.perf_counter()

    for round_num in range(args.num_rounds):
        print(f"--- Round {round_num + 1}/{args.num_rounds} ---")
        t_start = time.perf_counter()

        outputs = llm.generate(PROMPTS, sampling)

        t_end = time.perf_counter()
        round_time = t_end - t_start

        round_tokens = 0
        for i, out in enumerate(outputs):
            n = len(out.outputs[0].token_ids)
            text = out.outputs[0].text
            round_tokens += n
            print(f"  [{i}] {n:3d} tokens  |  {text[:80]}...")

        tok_per_sec = round_tokens / round_time if round_time > 0 else 0
        print(f"  Round {round_num + 1}: {round_tokens} tokens in {round_time:.2f}s  →  {tok_per_sec:.1f} tok/s\n")

        all_output_tokens += round_tokens
        latencies.append(round_time)

    total_time = time.perf_counter() - total_t0

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Total prompts:      {len(PROMPTS) * args.num_rounds}")
    print(f"  Max tokens/prompt:  {args.max_tokens}")
    print(f"  Total output tokens:{all_output_tokens}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Throughput:         {all_output_tokens / total_time:.1f} tok/s")
    print(f"  Avg round time:     {sum(latencies) / len(latencies):.2f}s")
    print(f"  Avg tokens/prompt:  {all_output_tokens / (len(PROMPTS) * args.num_rounds):.0f}")


if __name__ == "__main__":
    main()
