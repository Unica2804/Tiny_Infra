import asyncio
import logging
from contextlib import asynccontextmanager

import os

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer

from tinyinfra.api.router import router
from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.engine.loader import load_eos_token_ids, load_model
from tinyinfra.engine.scheduler import ContinuousBatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("tinyinfra.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing the Inference Server...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ref = os.environ.get("TINYINFRA_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    logger.info("Model: %s", model_ref)
    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    model_name = model_ref.rstrip("/").split("/")[-1]
    loaded = load_model(model_ref, device=device)
    model, config = loaded.model, loaded.config

    kv_cache = KVCacheManager(
        max_batch_size=8,
        max_seq_len=1024,
        num_layers=config.num_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        block_size=16,
        device=torch.device(device),
        dtype=torch.float16,
    )

    eos_ids = load_eos_token_ids(loaded.model_dir)
    eos_token_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id
    scheduler = ContinuousBatcher(model, kv_cache, eos_token_id=eos_token_id)
    background_task = asyncio.create_task(scheduler.run_loop())

    app.state.tokenizer = tokenizer
    app.state.scheduler = scheduler
    app.state.kv_cache = kv_cache
    app.state.background_loop_task = background_task
    app.state.model = model
    app.state.model_name = model_name

    logger.info("Inference Server initialized successfully.")
    yield

    logger.info("Shutting down the Inference Server...")
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    logger.info("Inference Server shut down successfully.")


app = FastAPI(title="Qwen Inference Server", lifespan=lifespan)
app.include_router(router)


def run() -> None:
    """Console entry point for `tinyinfra-serve`."""
    import os

    import uvicorn

    host = os.environ.get("TINYINFRA_HOST", "127.0.0.1")
    port = int(os.environ.get("TINYINFRA_PORT", "8000"))
    uvicorn.run("tinyinfra.api.server:app", host=host, port=port)
