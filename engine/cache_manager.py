# This file Initializes the cache manager for the application. It sets up the necessary configurations and ensures that the cache is ready for use.
import torch

class KVCacheManager:
    def __init__(self,
    max_batch_size: int,
    max_seq_len: int,
    num_layers:int,
    num_kv_heads:int,
    head_dim:int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        print("Allocating Kv cache on device: ", self.device)

        # Initialize the 5D cache shape based on the provided parameters
        cache_shape = (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim)

        #Preallocate the key_cache tensor with zeros and move it to the specified device
        self.key_cache = torch.zeros(cache_shape, device= device, dtype=dtype)
        # Preallocate the value_cache tensor with zeros and move it to the specified device
        self.value_cache = torch.zeros(cache_shape, device= device, dtype=dtype)

        # Create a 1D tensor to track the current sequence length for each batch slot this acts as the index to
        # the next position to write in the cache for each batch slot
        self.seq_len = torch.zeros(max_batch_size, device= device, dtype=torch.int32) 
        # Calculate the total bytes allocated for the key and value caches
        total_bytes = self.key_cache.element_size() * self.key_cache.nelement() * 2
        print(f"Allocated KV cache with shape {cache_shape} and total bytes: {total_bytes / (1024 ** 2):.2f} MB")

    # Method to append new tokens to the already allocated cache.
    def update_and_fetch(self, layer_idx: int, 
    batch_indices: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # validate the incoming batch indices tensor is not empty
        if batch_indices.numel() == 0:
            raise ValueError("batch_indices cannot be empty.")
        # Check if the incoming seq_len exceeds the maximum sequence length for any batch index 
        if torch.any(self.seq_len[batch_indices] > self.max_seq_len):
            raise ValueError("Sequence length exceeds maximum allowed length.")

        # If the input is 3D [Batch, Heads, Head_Dim], auto-expand it to 4D [Batch, Heads, 1, Head_Dim]
        if new_keys.ndim == 3:
            new_keys = new_keys.unsqueeze(2)
            new_values = new_values.unsqueeze(2)

        # Extract the incoming sequence length for the specified batch indices
        incoming_seq_len = new_keys.size(2)
        # grab the current sequence lengths for the specic user batch indices
        current_seq_len = self.seq_len[batch_indices]

        # iterate through each users request active in the current batch_indices

        for i, batch_idx in enumerate(batch_indices):
            # Calculate the start and end positions for the new keys and values in the cache
            start_pos = current_seq_len[i].item()
            end_pos = start_pos + incoming_seq_len

            # Check if the new data fits within the allocated cache size
            if end_pos > self.max_seq_len:
                raise ValueError(f"New data exceeds maximum sequence length for batch index {batch_idx}.")

            # Slice the pre allocated cache tensors to insert the new keys and values at the correct positions
            self.key_cache[layer_idx, batch_idx, :, start_pos:end_pos, :] = new_keys[i]
            self.value_cache[layer_idx, batch_idx, :, start_pos:end_pos, :] = new_values[i]
        # Update the sequence length tracker at the start of the layer.
        if layer_idx == 0:
            self.seq_len[batch_indices] += incoming_seq_len

        # Extract the max seq_len currently in the batch
        max_current_seq_len = torch.max(self.seq_len[batch_indices]).item()
        # Extract and return the active portion of the cache for the specified layer and batch indices
        fetched_keys = self.key_cache[layer_idx,batch_indices,:, :max_current_seq_len, :]
        fetched_values = self.value_cache[layer_idx,batch_indices,:, :max_current_seq_len, :]

        return fetched_keys, fetched_values
    
    # Method to clear the cache in a batch slot after generation is complete. 
    def free_batch_slot(self, batch_idx: int) -> None:
        # valaidate the batch index is within the valid range
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"batch_idx {batch_idx} is out of range.")
        self.seq_len[batch_idx] = 0