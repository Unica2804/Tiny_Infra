import asyncio
import time
from engine.model import QwenConfig, Qwen2ForCausalLM
from engine.cache_manager import KVCacheManager
from engine.scheduler import ContinuousBatcher

# Define a simulated API endpoint that an external user would hit
async def simulate_user_request(user_id: int, batcher: ContinuousBatcher, prompt: list[int], max_tokens: int):
    print(f"🟢 User {user_id} connected. Requesting {max_tokens} tokens...")
    start_time = time.perf_counter()
    
    # Send the request to our asynchronous batcher queue
    result = await batcher.generate(prompt_tokens=prompt, max_new_tokens=max_tokens)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    # Calculate tokens per second to measure our engine's throughput
    tps = len(result) / duration
    
    print(f"🔴 User {user_id} finished! Generated {len(result)} tokens in {duration:.2f}s ({tps:.2f} tokens/sec).")

async def main():
    print("⚙️ Initializing Engine for Stress Test...")
    # Load the basic configuration and empty model shell for CPU testing
    config = QwenConfig()
    model = Qwen2ForCausalLM(config)
    
    # Initialize the memory manager. Notice max_batch_size is 4.
    # We will send 6 users, forcing the engine to queue and swap them dynamically!
    kv_cache = KVCacheManager(
        max_batch_size=4, 
        max_seq_len=128, 
        num_layers=config.num_layers, 
        num_kv_heads=config.num_key_value_heads, 
        head_dim=config.hidden_size // config.num_attention_heads
    )
    
    # Boot up the continuous batcher
    batcher = ContinuousBatcher(model, kv_cache)
    
    # Launch the infinite background engine loop as an independent task
    engine_task = asyncio.create_task(batcher.run_loop())
    
    # Give the engine a fraction of a second to spin up
    await asyncio.sleep(0.1)
    
    print("\n🚀 Launching 6 concurrent user requests...")
    
    # Create 6 simulated users with wildly different token demands
    # Users 0-3 will grab the first 4 slots. Users 4 and 5 must wait in the queue.
    users = [
        simulate_user_request(0, batcher, [101, 102], max_tokens=10),
        simulate_user_request(1, batcher, [201, 202], max_tokens=50),
        simulate_user_request(2, batcher, [301, 302], max_tokens=15),
        simulate_user_request(3, batcher, [401, 402], max_tokens=30),
        simulate_user_request(4, batcher, [501, 502], max_tokens=20),
        simulate_user_request(5, batcher, [601, 602], max_tokens=40),
    ]
    
    # Run all users concurrently and wait for all of them to finish
    await asyncio.gather(*users)
    
    print("\n✅ All 6 users processed successfully. Stress test complete!")
    # Shut down the background engine loop
    engine_task.cancel()


asyncio.run(main())