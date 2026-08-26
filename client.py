import sys
import json
import asyncio
import httpx
from typing import List, Dict

SERVER_URL = "http://localhost:8000"

def prompt_param(
    label:str,
    default: float,
    cast_type = float
) -> float:
    user_val = input(f"{label} [{default}]: ").strip()
    if not user_val:
        return default
    try:
        return cast_type(user_val)
    except ValueError:
        print(f"Invalid input, using default {default}")
        return default

async def main():
    print("==================================================")
    print("                   Tiny Infra TUI               ")
    print("==================================================")

    async with httpx.AsyncClient(base_url = SERVER_URL, timeout=60.0) as client:
        try:
            response = await client.get("/info")
            response.raise_for_status()
            server_info = response.json()
        except Exception as e:
            print(f"Failed to fetch server info at {SERVER_URL}/info: {e}")
            return
        
        server_max_seq_len = server_info["max_seq_len"]
        print(f"Connected to server: {server_info['model_name']}")
        print(f"Server max sequence length: {server_max_seq_len}")
        print(f"Server max batch size: {server_info['max_batch_size']}")

        temperature = prompt_param("Temperature", 0.7, float)
        top_p = prompt_param("Top-P", 0.9, float)
        top_k = prompt_param("Top-K", 20, int)
        max_new_tokens = prompt_param("Max New Tokens", 256, int)

        budget = prompt_param("Conversation Budget (Tokens)", server_max_seq_len, int)
        conversation_budget = min(int(budget), server_max_seq_len)
        print(f"\nSession initialized with a conversation budget of {conversation_budget} tokens.\n")
        print("Type your message below. Commands: '/exit' to quit, '/clear' to clear the conversation.")

        #chat session state
        messages: List[Dict[str,str]] = []

        while True:
            try:
                user_input = input("\nUser> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if not user_input:
                continue
            if user_input.lower() == "/exit":
                print("Exiting...")
                break
            if user_input.lower() == "/clear":
                messages.clear()
                print("Conversation cleared.")
                continue
            messages.append({"role": "user", "content": user_input})

            # context budget warning
            total_chars = sum(len(msg["content"]) for msg in messages)
            estimated_tokens = int(total_chars / 3.5)
            if estimated_tokens + max_new_tokens >= conversation_budget:
                print(f"[Warning] Context history (~{estimated_tokens} tokens) is approaching your budget of {conversation_budget} tokens.")

            payload = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "p_value": top_p,
                "top_k": top_k,
                "stream": True
            }
            print("Assistant> ", end="", flush=True)
            accumulated_reply = []

            try:
                async with client.stream("POST", "/generate", json=payload) as response:
                    if response.status_code != 200:
                        error_detail = await response.aread()
                        print(f"\n[Error] Server returned status code {response.status_code}: {error_detail.decode()}")
                        messages.pop()  # Remove the last user message on error
                        continue
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if not raw_json:
                                continue
                            data = json.loads(raw_json)
                            if data.get("error"):
                                print(f"\n[Error] {data['error']}")
                                break
                            chunk = data.get("text", "")
                            if chunk:
                                sys.stdout.write(chunk)
                                sys.stdout.flush()
                                accumulated_reply.append(chunk)
                            if data.get("done"):
                                break
                full_reply = "".join(accumulated_reply)
                if full_reply:
                    messages.append({"role": "assistant", "content": full_reply})
                print()    
            except KeyboardInterrupt:
                print("\n\n [Stream Interrupted by User]")
                if accumulated_reply:
                    messages.append({"role": "assistant", "content": "".join(accumulated_reply)})
            except Exception as e:
                print(f"\n Communication failure: {e}")
                messages.pop()  
if __name__ == "__main__":
    asyncio.run(main())