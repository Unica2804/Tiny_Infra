from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Raw text prompt from the user")
    max_new_tokens: int = Field(default=128, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    p_value: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-P nucleus sampling threshold")
    top_k: int = Field(default=20, ge=0, description="Top-K token cutoff")
    stream: bool = Field(default=True, description="Whether to stream response via Server-Sent Events")

class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    prompt_tokens: int