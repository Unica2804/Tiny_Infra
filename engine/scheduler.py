# This is responsible for scheduling the inference tasks across available resources.

import asyncio
import torch
from typing import List, Optional
from models.qwen_25 import Qwen2ForCausalLM
from engine.cache_manager import KVCacheManager

# Class to hold user generation request information
class GenerationRequest:
    def __init__(self, prompt_tokens:List[int], max_new_tokens:int):
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
        self.completion_event = asyncio.Event()
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
        
    async def generate(self, prompt_tokens:List[int], max_new_tokens:int)-> List[int]:
        # Create a new generation request
        request = GenerationRequest(prompt_tokens, max_new_tokens)
        # Add the request to the queue
        await self.requests_queue.put(request)
        # Wait for the completion event to be set, indicating that the generation is complete
        await request.completion_event.wait()
        # return the generated tokens after the generation is complete
        return request.generated_tokens
    
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
                # Append the slot index to the list of active indices
                # active_indices.append(slot_idx)
                # Check if the req is new or does it needs processing
                if not req.is_prefilled:
                    prefill_indices.append(slot_idx) 
                else:
                    decode_indices.append(slot_idx) 

        # phase-1 prefill
        # process prefill requests individually
        for slot_idx in prefill_indices:
            self._execute_prefill(slot_idx)
        
        # phase-2 decode
        # we batch all decode requests together for efficiency to maximize paged_attn
        if decode_indices:
            self._execute_decode(decode_indices)
    
    def _execute_prefill(self, slot_idx:int):
        req = self.active_slots[slot_idx]

        # feed the entire prompt into the model
        input_idx = torch.tensor([req.prompt_tokens], device=self.device, dtype=torch.int32)
        batch_indices = torch.tensor([slot_idx], device=self.device, dtype=torch.int32) 

        with torch.no_grad():
            logits = self.model(input_idx, batch_indices, self.kv_cache, self.cos_cache, self.sin_cache)

        self.kv_cache.seq_len[batch_indices] += input_idx.size(1)
        # grab the prediction for the final token in the prompt
        next_token = torch.argmax(logits[:, -1, :],dim=-1).item()
        req.generated_tokens.append(next_token)

        hit_eos = (self.eos_token_id is not None) and (next_token == self.eos_token_id)
        hit_max_tokens = len(req.generated_tokens) >= req.max_new_tokens

        if hit_eos or hit_max_tokens:
            stop_reason = "EOS Token" if hit_eos else "Max Tokens"
            print(f"Request in slot {slot_idx} completed immediately during prefill ({stop_reason}).")
            req.completion_event.set()
            
            # Wipe the fragment block tables
            self.kv_cache.free_batch_slot(slot_idx)
            self.active_slots[slot_idx] = None
        else:
            # Only flag for the decode phase if generation needs to continue
            req.is_prefilled = True

    def _execute_decode(self, decode_indices: List[int]):
        current_tokens = []
        for slot_idx in decode_indices:
            req = self.active_slots[slot_idx]
            # Get the last generated token for each active request
            current_tokens.append(req.generated_tokens[-1])
        
        # Batch them together into [Batch, 1] tensor
        input_idx = torch.tensor(current_tokens, device=self.device, dtype=torch.int32).view(-1,1)
        batch_indices = torch.tensor(decode_indices, device=self.device, dtype=torch.int32)

        with torch.no_grad():
            logits = self.model(input_idx, batch_indices, self.kv_cache, self.cos_cache, self.sin_cache)
        
        self.kv_cache.seq_len[batch_indices] += 1

        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()

        for i, slot_idx in enumerate(decode_indices):
            req = self.active_slots[slot_idx]
            req.generated_tokens.append(next_tokens[i])

            hit_eos = (self.eos_token_id is not None) and (next_tokens == self.eos_token_id)
            hit_max_tokens = len(req.generated_tokens) >= req.max_new_tokens

            if hit_eos or hit_max_tokens:
                # Mark the request as complete and set the completion event
                print(f"Request in slot {slot_idx} has completed generation.")
                req.completion_event.set()
                # dynamically wipe the fragment block tables
                self.kv_cache.free_batch_slot(slot_idx)
                self.active_slots[slot_idx] = None
