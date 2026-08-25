import asyncio
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import AutoTokenizer
from models.config import QwenConfig # type: ignore
from engine.loader import load_qwen 
from engine.scheduler import ContinuousBatcher
from engine.cache_manager import KVCacheManager
from api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("inference_engine")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing the Inference Server...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_path = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_path = "weights/qwen25_0.5b_extracted.safetensors"
    model = load_qwen(model_path, device=device).to(torch.float16)
    config = QwenConfig()

    kv_cache = KVCacheManager(
        max_batch_size=8,
        max_seq_len=1024,
        num_layers=config.num_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        block_size=16,
        device=torch.device(device),
        dtype=torch.float16
    )

    scheduler = ContinuousBatcher(
        model,
        kv_cache,
        eos_token_id=tokenizer.eos_token_id
    )
    background_task = asyncio.create_task(scheduler.run_loop())

    app.state.tokenizer = tokenizer
    app.state.scheduler = scheduler
    app.state.kv_cache = kv_cache
    app.state.background_loop_task = background_task
    app.state.model = model
    
    logger.info("Inference Server initialized successfully.")
    yield

    logger.info("Shutting down the Inference Server...")
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    logger.info("Inference Server shut down successfully.")

app = FastAPI(
    title="Qwen Inference Server",
    lifespan = lifespan
)
app.include_router(router)