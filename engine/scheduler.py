# This is responsible for scheduling the inference tasks across available resources.

import asyncio
import torch
from typing import List, Optional
from engine.model import Qwen2ForCausalLM
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
    def __init__(self, model:Qwen2ForCausalLM, kv_cache:KVCacheManager):
        self.model = model
        self.kv_cache = kv_cache
        self.requests_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self.max_batch_size = kv_cache.max_batch_size
        # List to represent the physical cache slots, initialized with zeroes
        self.active_slots: List[Optional[GenerationRequest]] = [None] * self.max_batch_size
        self.device = kv_cache.device
    
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
        active_indices = []
        current_tokens = []
        # iterate through the active slots to gather the current tokens for each active request
        for slot_idx, req in enumerate(self.active_slots):
            # Check if the slot has an active request
            if req is not None:
                # Append the slot index to the list of active indices
                active_indices.append(slot_idx)
                # Check if the req is new or does it needs processing
                if not req.is_prefilled:
                    # !!! Modify this part later cuz we need to process the whole prompt rather than last token.

                    token_to_feed = req.prompt_tokens[-1] 
                    # Mark it as processed
                    req.is_prefilled = True
                else:
                    # If it's already prefilled, we only feed the last generated token
                    token_to_feed = req.generated_tokens[-1] 
                current_tokens.append(token_to_feed)
        # Convert the lists to tensors for model input
        batch_indices = torch.tensor(active_indices, device=self.device, dtype=torch.long)
        input_ids = torch.tensor(current_tokens, device=self.device, dtype=torch.long).view(-1,1)

        with torch.no_grad():
            logits = self.model(input_ids,batch_indices,self.kv_cache)
        # Extract the logits of the last token
        next_token_logits = logits[:, -1, :]
        # Use argmax to select the next token for each active request
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        # Convert it back to a list
        predicted_tokens_ids = next_tokens.tolist()

        # Iterate through the predicted tokens to map them back to their requests
        for i, slot_idx in enumerate(active_indices):
            # Grab the corresponding request from the active slots
            req = self.active_slots[slot_idx]
            # Grap the token Id that the model predicted for this request
            new_token = predicted_tokens_ids[i]
            # append the new token to the generated tokens list of the request
            req.generated_tokens.append(new_token)
        # Check if the user has reached the max_new_tokens limit, if so we can mark the request as complete and free up the slot
        if len(req.generated_tokens)>=req.max_new_tokens:
            print(f"Request in slot {slot_idx} has completed generation. Freeing up the slot.")
            # Trigger the completion event to notify the waiting coroutine that generation is complete
            req.completion_event.set()
            # Free the batch slot in the cache manager
            self.kv_cache.free_batch_slot(slot_idx)
            # set the active slot to None to indicate it's now free
            self.active_slots[slot_idx] = None

