from typing import Dict, Type
import logging
from core.config import LLMModelConfig, LLMUnifiedResponse, LLMExecutionMetrics
from core.wrappers import BaseLLMWrapper, OpenAIWrapper, GoogleGenAIWrapper, AWSBedrockWrapper

logger = logging.getLogger("MultiLLMFactory")

class LLMModelFactory:
    """Central Factory pattern engine handling provider lifecycle initialization configurations."""
    
    _registry: Dict[str, Type[BaseLLMWrapper]] = {
        "openai": OpenAIWrapper,
        "google": GoogleGenAIWrapper,
        "aws_bedrock": AWSBedrockWrapper
    }

    @classmethod
    def register_provider(cls, name: str, wrapper_cls: Type[BaseLLMWrapper]) -> None:
        """Pluggable registry engine hook allowing dynamic developer extension overrides."""
        cls._registry[name.lower()] = wrapper_cls
        logger.info(f"Dynamically appended custom routing strategy for mapping: '{name}'")

    @classmethod
    def create_client(cls, config: LLMModelConfig) -> BaseLLMWrapper:
        """Instantiates and validates Strategy classes mapping back to configurations rules."""
        provider_key = config.provider.lower()
        if provider_key not in cls._registry:
            raise ValueError(
                f"Requested driver provider identification key '{config.provider}' is missing from mapping registry. "
                f"Valid active runtime strategies: {list(cls._registry.keys())}"
            )
        return cls._registry[provider_key](config)

# Expose interfaces as a clean module bundle API footprint
__all__ = ["LLMModelFactory", "LLMModelConfig", "LLMUnifiedResponse", "BaseLLMWrapper"]