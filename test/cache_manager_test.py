import torch
from engine.cache_manager import KVCacheManager

# Wrap the test in a no_grad block since we are strictly doing inference memory management
with torch.no_grad():
        # Initialize a micro-cache matching Qwen2.5-0.5B architecture for a 4-user batch
    cache = KVCacheManager(
        max_batch_size=4,
        max_seq_len=1024,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
    )
        
    # Simulate an incoming generation step for two active users (Batch ID 0 and Batch ID 2)
    active_users = torch.tensor([0, 2], dtype=torch.long)
        
    # Simulate generating 1 new token's worth of Key vectors for these two users
    dummy_new_keys = torch.randn(2, 1, 2, 64)
    # Simulate generating 1 new token's worth of Value vectors for these two users
    dummy_new_values = torch.randn(2, 1, 2, 64)
        
    # Push the new tokens into Layer 0's cache and retrieve the history
    k_out, v_out = cache.update_and_fetch(0, active_users, dummy_new_keys, dummy_new_values)
        
    # Print a success message confirming the logic handles the pointers correctly
    print(f"✅ Successfully appended tokens! Retrieved cache shape: {k_out.shape}")