# This is responsible for scheduling the inference tasks across available resources.

import asyncio
import torch
from typing import List, Optional, AsyncGenerator
from models.qwen_25 import Qwen2ForCausalLM
from engine.cache_manager import KVCacheManager

_END_OF_STREAM = object()  # Sentinel value to indicate the end of a stream
# Class to hold user generation request information
class GenerationRequest:
    def __init__(self, prompt_tokens:List[int], max_new_tokens:int, temperature:float = 0.7, p_value: float = 0.9, top_k:int = 20):
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.p_value = p_value
        self.top_k = top_k
        self.generated_tokens = []
        self.token_queue = asyncio.Queue()
        self.completion_event = asyncio.Event()
        self.error = None
        self.is_prefilled = False

class ContinuousBatcher:
    def __init__(self, model:Qwen2ForCausalLM, kv_cache:KVCacheManager, eos_token_id: Optional[int]=None):
        self.model = model
        self.kv_cache = kv_cache
        self.requests_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self.max_batch_size = kv_cache.max_batch_size
        # List to represent the physical cache slots, initialized with zeroes
        self.active_slots: List[Optional[GenerationRequest]] = [None] * self.max_batch_size
        self.device = kv_cache.device
        self.cos_cache, self.sin_cache = self._init_rope_cache()
        self.eos_token_id = eos_token_id
    
    def _init_rope_cache(self):
        # precompute the RoPE cache for the model and store it in the KVCacheManager
        max_seq_len = self.kv_cache.max_seq_len
        head_dim = self.kv_cache.head_dim

        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        t = torch.arange(max_seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Compute the sine and cosine values for the RoPE cache
        cos = freqs.cos().to(torch.float32)
        sin = freqs.sin().to(torch.float32)
        return cos, sin
    
    async def generate_stream(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 0.7,
        p_value: float = 0.9,
        top_k: int = 20
    ) -> AsyncGenerator[int, None]:

        # Guard against exceeding the maximum sequence length
        if len(prompt_tokens) >= self.kv_cache.max_seq_len:
            raise ValueError(
                f"Prompt length ({len(prompt_tokens)}) exceeds or matches maximum "
                f"allowed sequence length ({self.kv_cache.max_seq_len})."
            )

        # Create a new generation request
        request = GenerationRequest(prompt_tokens, max_new_tokens, temperature, p_value, top_k)
        # Add the request to the queue
        await self.requests_queue.put(request)

        while True:
            # Wait for the next token to be available in the token queue
            next_token = await request.token_queue.get()
            if next_token is _END_OF_STREAM:
                if request.error is not None:
                    raise request.error
                break
            yield next_token


    async def generate(self, prompt_tokens:List[int], max_new_tokens:int, temperature:float = 0.7, p_value: float = 0.9, top_k:int = 20)-> List[int]:
        tokens = []
        async for token in self.generate_stream(
            prompt_tokens, max_new_tokens, temperature, p_value, top_k
        ):
            tokens.append(token)
        return tokens

    async def run_loop(self):
        print("Continious batcher engine has started and monitors the requests queue for incoming generation requests.")
        while True:
            # Try to fill the active slots with requests from the queue
            self._fill_active_slots()

            #Check if there are any active requests to process
            if self._has_active_requests():
                # Trigger a single generation step through the model
                self._step_generation()
            else:
                # If there is no active requests briefly yield control to avoid cpu cycles
                await asyncio.sleep(0.01)
            # Yield control to the event loop to allow generation requests again.
            await asyncio.sleep(0)
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        requests: List[GenerationRequest]) -> List[int]:

        # Applies Temperature, Top-K, and Top-P sampling to the logits for each request in the batch and returns the sampled next token IDs.
        next_token_logits = logits[:, -1, :].clone()
        batch_size, vocab_size = next_token_logits.shape
        next_tokens = []

        temps = torch.tensor([[req.temperature] for req in requests], device=self.device, dtype=logits.dtype) 
        
        safe_temps = torch.where(temps == 0.0, 1.0, temps)
        next_token_logits = next_token_logits / safe_temps

        top_k_values = [min(req.top_k, vocab_size) if req.top_k > 0 else vocab_size for req in requests]   
        max_k = max(top_k_values)

        # Top-K sampling
        if max_k < vocab_size:
            top_k_logits, _ = torch.topk(next_token_logits, max_k, dim=-1)
            # Create row mask for varying top_k parameters per request
            k_mask = torch.arange(max_k, device=self.device).unsqueeze(0) < torch.tensor(top_k_values, device=self.device).unsqueeze(1)
            # Replace masked out top_k positions with -inf
            cutoff_per_row = torch.full((batch_size, 1), float('-inf'), device=self.device, dtype=logits.dtype)
            cutoff_per_row = torch.where(k_mask, top_k_logits, cutoff_per_row).min(dim=-1, keepdim=True).values
            next_token_logits[next_token_logits < cutoff_per_row] = float('-inf')
            
        # P_value (nucleus) sampling
        p_values = torch.tensor([req.p_value for req in requests], device=self.device, dtype=logits.dtype)
        if (p_values < 1.0).any():
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Shift the cumulative probabilities to the right to keep the first token above threshold
            sorted_indices_to_remove = cumulative_probs > p_values.unsqueeze(1)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # Scatter mask back to the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
        
        #Greedy vs. Multinomial Sampling
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Batched multinomial sampling
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        greedy_tokens = torch.argmax(next_token_logits, dim=-1)
        is_greedy = (temps.squeeze(-1) == 0.0)
        final_tokens = torch.where(is_greedy, greedy_tokens, sampled_tokens)
        return final_tokens.tolist()

    # helper to pull requests from the queue and fill the active slots
    def _fill_active_slots(self):
        for i in range(self.max_batch_size):
            # Check if the current slot is empty and if there are pending requests in the queue
            if self.active_slots[i] is None and not self.requests_queue.empty():
                # Retrievve a new request from queue without blocking the event loop
                new_request = self.requests_queue.get_nowait()
                # Assign the new request to the active slot
                self.active_slots[i] = new_request
                print(f"Assigned new request to Cache slot {i}")
    
    # helper to check if there are any active requests in the slots
    def _has_active_requests(self) -> bool:
        # Check if any of the active slots contain a request (i.e., are not None)
        return any(req is not None for req in self.active_slots)
    
    # Main method to calculate next token for all active requests
    def _step_generation(self):
        # Create lists to hold active slots and corresponding tokens every tick
        prefill_indices = []
        decode_indices = []
        # iterate through the active slots to gather the current tokens for each active request
        for slot_idx, req in enumerate(self.active_slots):
            # Check if the slot has an active request
            if req is not None:
                
                # Check if the req is new or does it needs processing
                if not req.is_prefilled:
                    prefill_indices.append(slot_idx) 
                else:
                    decode_indices.append(slot_idx) 

        # phase-1 prefill
        # process prefill requests individually
        for slot_idx in prefill_indices:
            try:
                self._execute_prefill(slot_idx)
            except Exception as e:
                self._finish_request(slot_idx, reason="Error during prefill", error=e)
            
        
        # phase-2 decode
        # we batch all decode requests together for efficiency to maximize paged_attn
        if decode_indices:
            try:
                self._execute_decode(decode_indices)
            except Exception as e:
                for slot_idx in decode_indices:
                    self._finish_request(slot_idx, reason="Error during decode", error=e)

    def _finish_request(self, slot_idx:int, reason:str, error: Optional[Exception]=None):
        req = self.active_slots[slot_idx]
        if req is None:
            return
        req.error = error
        req.completion_event.set()
        req.token_queue.put_nowait(_END_OF_STREAM)

        self.kv_cache.free_batch_slot(slot_idx)
        self.active_slots[slot_idx] = None

        if error:
            print(f"Slot {slot_idx} request terminated due to error: {error}")
        else:
            print(f"Slot {slot_idx} request completed successfully due to: {reason}")

    def _execute_prefill(self, slot_idx:int):
        req = self.active_slots[slot_idx]

        # feed the entire prompt into the model
        input_idx = torch.tensor([req.prompt_tokens], device=self.device, dtype=torch.int32)
        batch_indices = torch.tensor([slot_idx], device=self.device, dtype=torch.int32) 

        with torch.no_grad():
            logits = self.model(input_idx, batch_indices, self.kv_cache, self.cos_cache, self.sin_cache)

        self.kv_cache.seq_len[batch_indices] += input_idx.size(1)
        # grab the prediction for the final token in the prompt
        next_token = self._sample_next_token(logits, [req])[0]
        req.generated_tokens.append(next_token)
        req.token_queue.put_nowait(next_token)

        hit_eos = (self.eos_token_id is not None) and (next_token == self.eos_token_id)
        hit_max_tokens = len(req.generated_tokens) >= req.max_new_tokens

        if hit_eos or hit_max_tokens:
            stop_reason = "EOS Token" if hit_eos else "Max Tokens"
            self._finish_request(slot_idx, reason=f"prefill ({stop_reason})")
        else:
            # Only flag for the decode phase if generation needs to continue
            req.is_prefilled = True

    def _execute_decode(self, decode_indices: List[int]):
        current_tokens = []
        active_requests = []
        for slot_idx in decode_indices:
            req = self.active_slots[slot_idx]
            # Get the last generated token for each active request
            current_tokens.append(req.generated_tokens[-1])
            active_requests.append(req)
        
        # Batch them together into [Batch, 1] tensor
        input_idx = torch.tensor(current_tokens, device=self.device, dtype=torch.int32).view(-1,1)
        batch_indices = torch.tensor(decode_indices, device=self.device, dtype=torch.int32)

        with torch.no_grad():
            logits = self.model(input_idx, batch_indices, self.kv_cache, self.cos_cache, self.sin_cache)
        
        self.kv_cache.seq_len[batch_indices] += 1

        next_tokens = self._sample_next_token(logits, active_requests)

        for i, slot_idx in enumerate(decode_indices):
            req = self.active_slots[slot_idx]
            new_token = next_tokens[i]
            req.generated_tokens.append(new_token)
            req.token_queue.put_nowait(new_token)

            hit_eos = (self.eos_token_id is not None) and (new_token == self.eos_token_id)
            hit_max_tokens = len(req.generated_tokens) >= req.max_new_tokens

            if hit_eos or hit_max_tokens:
                stop_reason = "EOS Token" if hit_eos else "Max Tokens"
                self._finish_request(slot_idx, reason=stop_reason)
