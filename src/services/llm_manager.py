"""
QCM Generator Pro - LLM Manager Service

This module handles interaction with various LLM providers including
local models (Ollama) and cloud APIs (OpenAI, Anthropic).
"""

import json
import logging
import time
from typing import Any, Optional

import httpx

from ..core.config import settings
from ..models.enums import ModelType
from .langsmith_tracker import get_langsmith_tracker

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback decorator
    def traceable(name: str = None, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised when LLM operations fail."""
    pass


class LLMResponse:
    """Represents a response from an LLM."""
    
    def __init__(
        self, 
        content: str, 
        model: str, 
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        return self.content


class LLMManager:
    """
    Manager for interacting with different LLM providers.
    """
    
    def __init__(self):
        """Initialize LLM manager."""
        self.default_model = settings.llm.default_model
        self.langsmith_tracker = get_langsmith_tracker()
        self.model_type = settings.llm.model_type
        
    async def call_openai_api(
        self, 
        prompt: str, 
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Call OpenAI API.
        
        Args:
            prompt: User prompt
            model: Model to use
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If API call fails
        """
        if not settings.llm.openai_api_key:
            raise LLMError("OpenAI API key not configured")
            
        model = model or settings.llm.openai_model
        temperature = temperature or settings.llm.default_temperature
        max_tokens = max_tokens or settings.llm.default_max_tokens
        
        headers = {
            "Authorization": f"Bearer {settings.llm.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=settings.llm.openai_timeout) as client:
                response = await client.post(
                    f"{settings.llm.openai_api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
            result = response.json()
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                model=model,
                usage=result.get("usage", {}),
                metadata={"provider": "openai", "response_id": result.get("id")}
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise LLMError(f"OpenAI API request failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise LLMError(f"Failed to connect to OpenAI API: {e}")
        except KeyError as e:
            logger.error(f"Unexpected OpenAI API response format: {e}")
            raise LLMError(f"Invalid OpenAI API response: {e}")
            
    async def call_anthropic_api(
        self, 
        prompt: str, 
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Call Anthropic API.
        
        Args:
            prompt: User prompt
            model: Model to use
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If API call fails
        """
        if not settings.llm.anthropic_api_key:
            raise LLMError("Anthropic API key not configured")
            
        model = model or settings.llm.anthropic_model
        temperature = temperature or settings.llm.default_temperature
        max_tokens = max_tokens or settings.llm.default_max_tokens
        
        headers = {
            "x-api-key": settings.llm.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            async with httpx.AsyncClient(timeout=settings.llm.anthropic_timeout) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
            result = response.json()
            
            return LLMResponse(
                content=result["content"][0]["text"],
                model=model,
                usage=result.get("usage", {}),
                metadata={"provider": "anthropic", "response_id": result.get("id")}
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise LLMError(f"Anthropic API request failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"Anthropic API connection error: {e}")
            raise LLMError(f"Failed to connect to Anthropic API: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Anthropic API response format: {e}")
            raise LLMError(f"Invalid Anthropic API response: {e}")
            
    async def call_ollama_api(
        self, 
        prompt: str, 
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Call Ollama local API.
        
        Args:
            prompt: User prompt
            model: Model to use
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If API call fails
        """
        if not settings.llm.ollama_base_url:
            raise LLMError("Ollama base URL not configured")
            
        model = model or self.default_model
        temperature = temperature or settings.llm.default_temperature
        max_tokens = max_tokens or settings.llm.default_max_tokens
        
        # Format prompt with system message if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=settings.llm.ollama_timeout) as client:
                response = await client.post(
                    f"{settings.llm.ollama_base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
            result = response.json()
            
            return LLMResponse(
                content=result["response"],
                model=model,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                metadata={
                    "provider": "ollama", 
                    "eval_duration": result.get("eval_duration"),
                    "load_duration": result.get("load_duration")
                }
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise LLMError(f"Ollama API request failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"Ollama API connection error: {e}")
            raise LLMError(f"Failed to connect to Ollama API: {e}")
        except KeyError as e:
            logger.error(f"Unexpected Ollama API response format: {e}")
            raise LLMError(f"Invalid Ollama API response: {e}")
            
    @traceable(name="llm_generate_response", run_type="llm")
    async def generate_response(
        self, 
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
        langsmith_run_id: Optional[str] = None,
        # LangSmith metadata
        examples_file: Optional[str] = None,
        max_examples: Optional[int] = None,
        topic: Optional[str] = None,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response using the configured LLM provider.
        
        Args:
            prompt: User prompt
            model: Model to use
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            provider: Specific provider to use
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If generation fails
        """
        logger.debug(f"Generating LLM response with prompt length: {len(prompt)}")
        
        # Log the Few-Shot Examples if present for debugging
        if examples_file:
            logger.info(f"Using Few-Shot Examples: {examples_file} (max: {max_examples})")
        
        start_time = time.time()
        response = None
        used_provider = None
        used_model = model or self.default_model
        
        try:
            # Determine provider
            if provider:
                if provider == "openai" and settings.llm.openai_api_key:
                    response = await self.call_openai_api(prompt, model, system_prompt, temperature, max_tokens)
                    used_provider = "openai"
                elif provider == "anthropic" and settings.llm.anthropic_api_key:
                    response = await self.call_anthropic_api(prompt, model, system_prompt, temperature, max_tokens)
                    used_provider = "anthropic"
                elif provider == "ollama" and settings.llm.ollama_base_url:
                    response = await self.call_ollama_api(prompt, model, system_prompt, temperature, max_tokens)
                    used_provider = "ollama"
                else:
                    raise LLMError(f"Provider {provider} not available or not configured")
                    
            else:
                # Auto-select provider based on configuration
                # Priority: OpenAI > Anthropic > Ollama
                if settings.llm.openai_api_key:
                    try:
                        response = await self.call_openai_api(prompt, model, system_prompt, temperature, max_tokens)
                        used_provider = "openai"
                    except LLMError as e:
                        logger.warning(f"OpenAI API failed, trying fallback: {e}")
                        
                if not response and settings.llm.anthropic_api_key:
                    try:
                        response = await self.call_anthropic_api(prompt, model, system_prompt, temperature, max_tokens)
                        used_provider = "anthropic"
                    except LLMError as e:
                        logger.warning(f"Anthropic API failed, trying fallback: {e}")
                        
                if not response and settings.llm.ollama_base_url:
                    try:
                        response = await self.call_ollama_api(prompt, model, system_prompt, temperature, max_tokens)
                        used_provider = "ollama"
                    except LLMError as e:
                        logger.warning(f"Ollama API failed: {e}")
                        
                if not response:
                    raise LLMError("No LLM provider available or configured")
            
            # Log successful LLM call (LangSmith tracing handled by @traceable decorator)
            generation_time = time.time() - start_time
            logger.info(f"LLM call successful - {used_provider}/{used_model} - {generation_time:.2f}s")
            
            # Enrich response metadata with generation context
            if response and hasattr(response, 'metadata'):
                response.metadata.update({
                    "generation_time": generation_time,
                    "provider": used_provider,
                    "model": used_model,
                    "topic": topic,
                    "question_type": question_type,
                    "difficulty": difficulty,
                    "examples_file": examples_file,
                    "max_examples": max_examples,
                    "uses_fewshot": examples_file is not None
                })
            
            return response
            
        except Exception as e:
            # Log error (LangSmith tracing handled by @traceable decorator)
            error_time = time.time() - start_time
            logger.error(f"LLM call failed after {error_time:.2f}s: {e}")
            raise
        
    async def test_connection(self, provider: str | None = None) -> dict[str, Any]:
        """
        Test connection to LLM provider.
        
        Args:
            provider: Specific provider to test
            
        Returns:
            Test results
        """
        results = {}
        
        providers_to_test = []
        if provider:
            providers_to_test = [provider]
        else:
            if settings.llm.openai_api_key:
                providers_to_test.append("openai")
            if settings.llm.anthropic_api_key:
                providers_to_test.append("anthropic")
            if settings.llm.ollama_base_url:
                providers_to_test.append("ollama")
                
        for prov in providers_to_test:
            try:
                response = await self.generate_response(
                    "Test message. Please respond with 'OK'.",
                    provider=prov,
                    max_tokens=10
                )
                results[prov] = {
                    "status": "success",
                    "model": response.model,
                    "response_length": len(response.content),
                    "usage": response.usage
                }
            except Exception as e:
                results[prov] = {
                    "status": "error",
                    "error": str(e)
                }
                
        return results
        
    async def list_available_models(self, provider: str | None = None) -> dict[str, list[str]]:
        """
        List available models for each provider.
        
        Args:
            provider: Specific provider
            
        Returns:
            Dictionary of provider -> models
        """
        models = {}
        
        # For Ollama, we can query the models endpoint
        if (not provider or provider == "ollama") and settings.llm.ollama_base_url:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
                    response.raise_for_status()
                    result = response.json()
                    models["ollama"] = [model["name"] for model in result.get("models", [])]
            except Exception as e:
                logger.warning(f"Failed to list Ollama models: {e}")
                models["ollama"] = [self.default_model]
                
        # For cloud providers, return known models
        if not provider or provider == "openai":
            models["openai"] = [
                "gpt-4", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-mini-16k"
            ]
            
        if not provider or provider == "anthropic":
            models["anthropic"] = [
                "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307", "claude-instant-1.2"
            ]
            
        return models


# Global LLM manager instance
_llm_manager: LLMManager | None = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


# Convenience functions
async def generate_llm_response(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    **kwargs
) -> LLMResponse:
    """
    Generate response using the LLM manager.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model to use
        **kwargs: Additional arguments
        
    Returns:
        LLM response
    """
    manager = get_llm_manager()
    return await manager.generate_response(
        prompt, 
        model=model,
        system_prompt=system_prompt,
        **kwargs
    )


async def test_llm_connection(provider: str | None = None) -> dict[str, Any]:
    """Test LLM provider connections."""
    manager = get_llm_manager()
    return await manager.test_connection(provider)


def generate_llm_response_sync(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    **kwargs
) -> LLMResponse:
    """
    Synchronous wrapper for LLM response generation.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model to use
        **kwargs: Additional arguments
        
    Returns:
        LLM response
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(
        generate_llm_response(prompt, system_prompt, model, **kwargs)
    )


def test_llm_connection_sync(provider: str | None = None) -> dict[str, Any]:
    """
    Synchronous wrapper for test_llm_connection.
    
    Args:
        provider: Provider to test (optional)
        
    Returns:
        Connection test results
    """
    import asyncio
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(test_llm_connection(provider))
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in sync LLM connection test: {e}")
        return {
            "error": {
                "status": "error",
                "error": str(e)
            }
        }