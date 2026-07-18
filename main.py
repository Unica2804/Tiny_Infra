import asyncio
import torch
from transformers import AutoTokenizer
from models.config import QwenConfig

from engine.loader import load_qwen
from engine.cache_manager import KVCacheManager
from engine.scheduler import ContinuousBatcher

async def run_inference():
    print("🚀 Booting Custom Inference Engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the Tokenizer (using HuggingFace just for text <-> ID conversion)
    tokenizer_path = "Qwen/Qwen2.5-0.5B" # Change to local path if you downloaded it
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. Load the Model Weights into our Custom Architecture
    model_path = "weights/qwen25_0.5b_extracted.safetensors"
    model = load_qwen(model_path, device=device)
    
    # FP16 conversion for our custom kernels
    model = model.to(torch.float16)

    config = QwenConfig()

    # 3. Initialize the Paged KV-Cache Manager
    kv_cache = KVCacheManager(
        max_batch_size=4,
        max_seq_len=1024,
        num_layers=config.num_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        block_size=16, 
        device=torch.device(device),
        dtype=torch.float16
    )

    eos_token_id = tokenizer.eos_token_id  # Get the EOS token ID from the tokenizer

    # 4. Initialize the Continuous Batching Scheduler
    scheduler = ContinuousBatcher(model, kv_cache, eos_token_id=eos_token_id)

    # Start the scheduler's background loop
    loop_task = asyncio.create_task(scheduler.run_loop())

    # 5. Define two concurrent user requests
    prompt_1 = "The capital of France is"
    prompt_2 = "Write a short poem about the ocean:"
    
    tokens_1 = tokenizer.encode(prompt_1)
    tokens_2 = tokenizer.encode(prompt_2)

    print("\n📩 Sending requests to the scheduler...")
    
    # Run both generation tasks simultaneously (Continuous Batching in action)
    task1 = scheduler.generate(prompt_tokens=tokens_1, max_new_tokens=300)
    task2 = scheduler.generate(prompt_tokens=tokens_2, max_new_tokens=500)

    # Wait for both to finish
    results = await asyncio.gather(task1, task2)

    # 6. Decode and Print the Results
    print("\n=========================================")
    print(f"User 1 Prompt: {prompt_1}")
    print(f"User 1 Output: {tokenizer.decode(results[0], skip_special_tokens=True)}")
    print("-----------------------------------------")
    print(f"User 2 Prompt: {prompt_2}")
    print(f"User 2 Output: {tokenizer.decode(results[1], skip_special_tokens=True)}")
    print("=========================================")

    # Kill the background loop once done
    loop_task.cancel()

if __name__ == "__main__":
    asyncio.run(run_inference())