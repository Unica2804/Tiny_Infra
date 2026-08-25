import json
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from api.schema import GenerationRequest, GenerationResponse

router = APIRouter()
logger = logging.getLogger("inference_engine")

@router.post("/generate", response_model=None)
async def generate_endpoint(payload: GenerationRequest, request: Request):
    tokenizer = request.app.state.tokenizer
    scheduler = request.app.state.scheduler
    kv_cache = request.app.state.kv_cache

    # Apply chat template for qwen
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": payload.prompt}],
        tokenize = False,
        add_generation_prompt = True
    )
    tokens = tokenizer.encode(formatted_prompt)

    # Error handling for context length
    if len(tokens) >= kv_cache.max_seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum context length of {kv_cache.max_seq_len}"
        )

    # Non streaming response
    if not payload.stream:
        try:
            output_tokens = await scheduler.generate(
                prompt_tokens=tokens,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                p_value=payload.p_value,
                top_k=payload.top_k
            )
            decoded_text = tokenizer.decode(
                output_tokens,
                skip_special_tokens=True
            )
            return GenerationResponse(
                text=decoded_text,
                tokens_generated=len(output_tokens),
                prompt_tokens=len(tokens)
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred during text generation."
            )
    
    # Streaming response
    async def event_stream():
        
        stream_gen = scheduler.generate_stream(
            prompt_tokens=tokens,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            p_value=payload.p_value,
            top_k=payload.top_k
        )
        generated_ids = []
        printed_len = 0
        try:
            async for token_id in stream_gen:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping generation.")
                    break
                generated_ids.append(token_id)
                full_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                new_text = full_text[printed_len:]
                if new_text:
                    printed_len = len(full_text)
                    chunk_data = json.dumps(
                        {"text": new_text, "done":False}
                    )
                    yield f"data: {chunk_data}\n\n"
            # Final chunk indicating completion
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        except Exception as e:
            logger.error(
                f"Error during streaming generation: {e}"
            )
            error_data = json.dumps(
                {"error": str(e), "done": True}
            )
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )