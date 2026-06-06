import abc
import json
import logging
import time
from typing import Any, Dict, Optional, Type
import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import LLMModelConfig, LLMUnifiedResponse, LLMExecutionMetrics

logger = logging.getLogger("MultiLLMEngine")

class BaseLLMWrapper(abc.ABC):
    """Abstract Strategy Contract enforcing structural alignment across provider SDKs."""
    
    def __init__(self, config: LLMModelConfig) -> None:
        self.config = config
        self.client = self._initialize_client()

    @abc.abstractmethod
    def _initialize_client(self) -> Any:
        """Internal bootstrap sequence logic hook for native client configuration mapping."""
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        """Execute standard synchronous chat completion query routing loops."""
        pass

    @abc.abstractmethod
    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        """Execute high-concurrency non-blocking event loop execution frames."""
        pass

    def _log_metrics(self, start_time: float, prompt_tokens: int, completion_tokens: int) -> LLMExecutionMetrics:
        """Calculates precise transaction execution timeline footprints."""
        latency = (time.perf_counter() - start_time) * 1000
        metrics = LLMExecutionMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency, 2)
        )
        logger.info(f"Target '{self.config.model_id}' query resolved. Time: {metrics.latency_ms}ms | Total Tokens: {metrics.total_tokens}")
        return metrics

# ==========================================
# OPENAI STRATEGY DRIVER
# ==========================================
class OpenAIWrapper(BaseLLMWrapper):
    def _initialize_client(self) -> Any:
        from openai import OpenAI, AsyncOpenAI
        api_str = self.config.api_key.get_secret_value() if self.config.api_key else None
        return {
            "sync": OpenAI(api_key=api_str, timeout=self.config.timeout),
            "async": AsyncOpenAI(api_key=api_str, timeout=self.config.timeout)
        }

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        start_time = time.perf_counter()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"response_format": response_schema} if response_schema else {}

        response = self.client["sync"].beta.chat.completions.parse(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )
        choice = response.choices[0].message
        metrics = self._log_metrics(start_time, response.usage.prompt_tokens, response.usage.completion_tokens)
        return LLMUnifiedResponse(
            content=choice.content or "",
            structured_json=choice.parsed.model_dump() if response_schema and choice.parsed else None,
            metrics=metrics
        )

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        start_time = time.perf_counter()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"response_format": response_schema} if response_schema else {}

        response = await self.client["async"].beta.chat.completions.parse(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )
        choice = response.choices[0].message
        metrics = self._log_metrics(start_time, response.usage.prompt_tokens, response.usage.completion_tokens)
        return LLMUnifiedResponse(
            content=choice.content or "",
            structured_json=choice.parsed.model_dump() if response_schema and choice.parsed else None,
            metrics=metrics
        )

# ==========================================
# GOOGLE GENAI STRATEGY DRIVER
# ==========================================
class GoogleGenAIWrapper(BaseLLMWrapper):
    def _initialize_client(self) -> Any:
        from google import genai
        api_str = self.config.api_key.get_secret_value() if self.config.api_key else None
        return genai.Client(api_key=api_str)

    def _build_config(self, system_instruction: Optional[str], response_schema: Optional[Type[Any]]) -> Any:
        from google.genai import types
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            system_instruction=system_instruction,
            response_mime_type="application/json" if response_schema else "text/plain",
            response_schema=response_schema if response_schema else None
        )

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        start_time = time.perf_counter()
        gen_config = self._build_config(system_instruction, response_schema)
        response = self.client.models.generate_content(
            model=self.config.model_id, contents=prompt, config=gen_config
        )
        p_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        c_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        metrics = self._log_metrics(start_time, p_tok, c_tok)
        struct_data = json.loads(response.text) if response_schema and response.text else None
        return LLMUnifiedResponse(content=response.text or "", structured_json=struct_data, metrics=metrics)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        start_time = time.perf_counter()
        gen_config = self._build_config(system_instruction, response_schema)
        response = await self.client.aio.models.generate_content(
            model=self.config.model_id, contents=prompt, config=gen_config
        )
        p_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        c_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        metrics = self._log_metrics(start_time, p_tok, c_tok)

# --- Separate Reasoning Streams From Core Content ---
        content_chunks = []
        reasoning_chunks = []
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                # Detect native thinking blocks via the SDK 'thought' attribute
                if getattr(part, 'thought', False):
                    reasoning_chunks.append(part.text)
                else:
                    content_chunks.append(part.text)

        # Build clean strings from collected chunks
        final_content = "".join(content_chunks) if content_chunks else (response.text or "")
        reasoning_text = "".join(reasoning_chunks) if reasoning_chunks else None
        
        # Parse structured JSON safely using clean content text
        struct_data = json.loads(final_content) if response_schema and final_content else None
        
        return LLMUnifiedResponse(
            content=final_content,
            structured_json=struct_data,
            metrics=metrics,
            reasoning_content=reasoning_text  # Passes the isolated thought track down the cascade
        )

# ==========================================
# AWS BEDROCK STRATEGY DRIVER
# ==========================================
class AWSBedrockWrapper(BaseLLMWrapper):
    def _initialize_client(self) -> Any:
        import boto3
        from botocore.config import Config
        botocore_cfg = Config(connect_timeout=self.config.timeout, read_timeout=self.config.timeout, retries={'max_attempts': 0})
        return boto3.client("bedrock-runtime", region_name=self.config.aws_region, config=botocore_cfg)

    def _prepare_payload(self, prompt: str, system_instruction: Optional[str], response_schema: Optional[Type[Any]]) -> Dict[str, Any]:
        payload = {
            "modelId": self.config.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": self.config.temperature,
                "maxTokens": self.config.max_tokens,
                "topP": self.config.top_p
            }
        }
        if system_instruction:
            payload["system"] = [{"text": system_instruction}]
        if response_schema:
            payload["additionalModelRequestFields"] = {
                "outputConfig": {"textFormat": {"json": response_schema.model_json_schema()}}
            }
        return payload

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        start_time = time.perf_counter()
        payload = self._prepare_payload(prompt, system_instruction, response_schema)
        response = self.client.converse(**payload)
        
        output_text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        metrics = self._log_metrics(start_time, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        struct_data = json.loads(output_text) if response_schema else None
        return LLMUnifiedResponse(content=output_text, structured_json=struct_data, metrics=metrics)

    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, response_schema: Optional[Type[Any]] = None) -> LLMUnifiedResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system_instruction, response_schema)