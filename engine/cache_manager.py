# This file Initializes the cache manager for the application. It sets up the necessary configurations and ensures that the cache is ready for use.
import torch

class KVCacheManager:
    def __init__(self,
    max_batch_size: int,
    max_seq_len: int,
    num_layers:int,
    num_kv_heads:int,
    head_dim:int,
    block_size: int =16,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        print("Allocating Kv cache on device: ", self.device)

        # Calculate block requirements based on the maximum sequence length and block size
        self.max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.total_physical_blocks = max_batch_size * self.max_blocks_per_seq

        # Initialize the 5D cache shape based on the provided parameters
        cache_shape = (num_layers, self.total_physical_blocks, num_kv_heads, block_size, head_dim)

        #Preallocate the key_cache tensor with zeros and move it to the specified device
        self.key_cache = torch.zeros(cache_shape, device= device, dtype=dtype)
        # Preallocate the value_cache tensor with zeros and move it to the specified device
        self.value_cache = torch.zeros(cache_shape, device= device, dtype=dtype)

        # Create a 1D tensor to track the current sequence length for each batch slot this acts as the index to
        # the next position to write in the cache for each batch slot
        self.seq_len = torch.zeros(max_batch_size, device= device, dtype=torch.int32)

        #Block tables are used to track the mapping of logical blocks to physical blocks in the cache.
        self.block_tables = torch.full((max_batch_size, self.max_blocks_per_seq), -1, dtype=torch.int32, device=device) 

        #List to track the next available physical block for each batch slot, initialized to 0
        self.free_blocks = list(range(self.total_physical_blocks))

        # Calculate the total bytes allocated for the key and value caches
        total_bytes = self.key_cache.element_size() * self.key_cache.nelement() * 2
        print(f"Allocated KV cache with shape {cache_shape} and total bytes: {total_bytes / (1024 ** 2):.2f} MB")

    # Method to append new tokens to the already allocated cache.
    def allocate_and_insert(self, layer_idx: int, 
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

        # iterate through each users request active in the current batch_indices

        for i, batch_idx in enumerate(batch_indices.tolist()):
            # Calculate the start and end positions for the new keys and values in the cache
            start_pos = self.seq_len[batch_idx].item()

            for t in range(incoming_seq_len):
                current_token_pos = start_pos + t
                if current_token_pos >= self.max_seq_len:
                    raise ValueError(f"Sequence length exceeds maximum allowed length for batch index {batch_idx}.")
                
                logical_block_idx = current_token_pos // self.block_size
                block_offset = current_token_pos % self.block_size

                # Check if physical block doesn't have a physical page yet, allocate one
                if self.block_tables[batch_idx, logical_block_idx] == -1:
                    if not self.free_blocks:
                        raise RuntimeError("OOM : No free physical blocks available for allocation.")

                    # pop a free block and assign it to the routing table
                    allocated_block = self.free_blocks.pop(0)
                    self.block_tables[batch_idx, logical_block_idx] = allocated_block

                # look up the physical block index
                physical_block = self.block_tables[batch_idx, logical_block_idx].item()

                # Slice the pre allocated cache tensors to insert the new keys and values at the correct positions
                self.key_cache[layer_idx, physical_block, :, block_offset, :] = new_keys[i, :, t, :]
                self.value_cache[layer_idx, physical_block, :, block_offset, :] = new_values[i, :, t, :]

        return self.block_tables , self.seq_len
    
    # Method to clear the cache in a batch slot after generation is complete. 
    def free_batch_slot(self, batch_idx: int) -> None:
        # valaidate the batch index is within the valid range
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"batch_idx {batch_idx} is out of range.")
        
        # Itterate through the routing table and free any assigned blocks
        for logical_block_idx in range(self.max_blocks_per_seq):
            physical_block = self.block_tables[batch_idx, logical_block_idx].item()

            if physical_block != -1:
                # Return to the available pool
                self.free_blocks.append(physical_block)
                # Reset the routing table entry
                self.block_tables[batch_idx, logical_block_idx] = -1
        
        # Reset the sequence length for the batch slot
        self.seq_len[batch_idx] = 0