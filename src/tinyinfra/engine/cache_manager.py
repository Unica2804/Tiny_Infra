# This file Initializes the cache manager for the application. It sets up the necessary configurations and ensures that the cache is ready for use.
import torch


class KVCacheManager:
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
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

        self.max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.total_physical_blocks = max_batch_size * self.max_blocks_per_seq
        cache_shape = (num_layers, self.total_physical_blocks, num_kv_heads, block_size, head_dim)
        self.key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.seq_len = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        self.block_tables = torch.full(
            (max_batch_size, self.max_blocks_per_seq), -1, dtype=torch.int32, device=device
        )
        
        self.free_blocks_t = torch.arange(self.total_physical_blocks, device=device, dtype=torch.int32)
        self.free_count = self.total_physical_blocks

        total_bytes = self.key_cache.element_size() * self.key_cache.nelement() * 2
        print(
            f"Allocated KV cache with shape {cache_shape} and total bytes: {total_bytes / (1024**2):.2f} MB"
        )

    def _insert_prefill(self, layer_idx, batch_indices, new_keys, new_values):
        """Vectorized path for multi-token prefill — removes per-token host syncs (TTFT)."""
        T = new_keys.size(2)
        B, H, D = new_keys.shape[0], new_keys.shape[1], new_keys.shape[3]
        starts = self.seq_len[batch_indices]                                    # [B]
        pos = starts[:, None] + torch.arange(T, device=self.device)             # [B,T]
        if (pos >= self.max_seq_len).any():
            raise ValueError("Sequence length exceeds maximum allowed length.")
        logical = pos // self.block_size
        offset = pos % self.block_size
        flat_idx = batch_indices[:, None].long() * self.max_blocks_per_seq + logical

        candidates = torch.unique(flat_idx)
        uniq = candidates[self.block_tables.view(-1)[candidates] == -1]
        n_new = uniq.numel()
        if n_new > self.free_count:
            raise RuntimeError("OOM: No free physical blocks available for allocation.")
        new_blocks = self.free_blocks_t[self.free_count - n_new : self.free_count]
        self.block_tables.view(-1)[uniq] = new_blocks
        self.free_count -= n_new

        physical = self.block_tables.view(-1)[flat_idx]                          # [B,T]
        NB, BS, NH = self.total_physical_blocks, self.block_size, self.num_kv_heads
        base = (layer_idx * NB + physical) * (NH * BS * D)                        # [B,T]
        heads = torch.arange(H, device=self.device).view(1, 1, H, 1) * (BS * D)
        offs = offset.view(B, T, 1, 1) * D
        dims = torch.arange(D, device=self.device).view(1, 1, 1, D)
        flat = base.view(B, T, 1, 1) + heads + offs + dims                       # [B,T,H,D]
        self.key_cache.view(-1)[flat] = new_keys.permute(0, 2, 1, 3)
        self.value_cache.view(-1)[flat] = new_values.permute(0, 2, 1, 3)

    def allocate_and_insert(
        self,
        layer_idx: int,
        batch_indices: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_indices.numel() == 0:
            raise ValueError("batch_indices cannot be empty.")
        if torch.any(self.seq_len[batch_indices] > self.max_seq_len):
            raise ValueError("Sequence length exceeds maximum allowed length.")
        if new_keys.ndim == 3:
            new_keys = new_keys.unsqueeze(2)
            new_values = new_values.unsqueeze(2)
        incoming_seq_len = new_keys.size(2)

        # DECODE (T=1)
        if incoming_seq_len == 1:
            for i, batch_idx in enumerate(batch_indices.tolist()):
                start_pos = self.seq_len[batch_idx].item()
                current_token_pos = start_pos
                if current_token_pos >= self.max_seq_len:
                    raise ValueError("Sequence length exceeds maximum allowed length.")
                logical_block_idx = current_token_pos // self.block_size
                block_offset = current_token_pos % self.block_size
                if self.block_tables[batch_idx, logical_block_idx] == -1:
                    if self.free_count <= 0:
                        raise RuntimeError("OOM: No free physical blocks available for allocation.")
                    block = int(self.free_blocks_t[self.free_count - 1].item())
                    self.block_tables[batch_idx, logical_block_idx] = block
                    self.free_count -= 1
                physical_block = self.block_tables[batch_idx, logical_block_idx].item()
                self.key_cache[layer_idx, physical_block, :, block_offset, :] = new_keys[i, :, 0, :]
                self.value_cache[layer_idx, physical_block, :, block_offset, :] = new_values[i, :, 0, :]
        else:
            self._insert_prefill(layer_idx, batch_indices, new_keys, new_values)

        return self.block_tables, self.seq_len

    def free_batch_slot(self, batch_idx: int) -> None:
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"batch_idx {batch_idx} is out of range.")
        row = self.block_tables[batch_idx]
        allocated = (row != -1).nonzero(as_tuple=False).flatten()
        if allocated.numel() > 0:
            ids = row[allocated]
            n = ids.numel()
            self.free_blocks_t[self.free_count : self.free_count + n] = ids
            self.free_count += n
            row[allocated] = -1
        self.seq_len[batch_idx] = 0
