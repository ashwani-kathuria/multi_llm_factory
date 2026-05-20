from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, SecretStr

class LLMModelConfig(BaseModel):
    """Centralized verification state engine for model parameters and targets."""
    provider: str = Field(..., description="Target provider system key (e.g., 'openai', 'google', 'aws_bedrock')")
    model_id: str = Field(..., description="Target model identifier tag")
    api_key: Optional[SecretStr] = Field(None, description="Optional raw credential pass-through")
    aws_region: Optional[str] = Field("us-east-1", description="Default regional target for AWS calls")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature setting")
    max_tokens: int = Field(2048, ge=1, description="Upper bound allocation context ceiling")
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    timeout: float = Field(30.0, description="Network circuit connection boundary timeout")

class LLMExecutionMetrics(BaseModel):
    """Normalized structured data schema container for latency and tokens usage telemetry."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

class LLMUnifiedResponse(BaseModel):
    """Normalized interface package returned back up to the user pipeline layer."""
    content: str
    structured_json: Optional[Dict[str, Any]] = None
    metrics: LLMExecutionMetrics