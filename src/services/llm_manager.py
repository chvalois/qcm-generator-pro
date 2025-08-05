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
            raise LLMError("OpenAI API key not configured (LLM_OPENAI_API_KEY environment variable)")
            
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
        
        # Check if model is available
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                models_response = await client.get(f"{settings.llm.ollama_base_url}/api/tags")
                if models_response.status_code == 200:
                    available_models = [m["name"] for m in models_response.json().get("models", [])]
                    if model not in available_models:
                        logger.warning(f"Model {model} not found in Ollama. Available: {available_models}")
                        raise LLMError(f"Model {model} not available in Ollama. Available models: {', '.join(available_models)}")
        except httpx.RequestError as e:
            logger.warning(f"Could not check Ollama models: {e}")
            # Continue anyway - maybe the model is available
            
        temperature = temperature or settings.llm.default_temperature
        max_tokens = max_tokens or settings.llm.default_max_tokens
        
        # Increase timeout for large models
        timeout = settings.llm.ollama_timeout
        if model and ("70b" in model.lower() or "72b" in model.lower()):
            timeout = min(timeout * 3, 600)  # 3x timeout for large models, max 10 minutes
        
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
            logger.info(f"Calling Ollama API with model: {model}, timeout: {timeout}s")
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{settings.llm.ollama_base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
            result = response.json()
            logger.debug(f"Ollama API response received for model: {model}")
            
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
            
    @traceable(
        name="llm_generate_response", 
        run_type="llm"
    )
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
        
        # Set up LangSmith tracking metadata before making the call
        try:
            from langsmith import get_current_run_tree
            current_run = get_current_run_tree()
            if current_run:
                # Set up initial metadata 
                current_run.extra = current_run.extra or {}
                current_run.extra.update({
                    "topic": topic,
                    "question_type": question_type,
                    "difficulty": difficulty,
                    "examples_file": examples_file,
                    "max_examples": max_examples,
                    "uses_fewshot": examples_file is not None
                })
                
        except ImportError:
            pass  # LangSmith not available
        except Exception as e:
            logger.debug(f"Failed to update initial LangSmith metadata: {e}")
        
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
            
            # Update LangSmith trace with final provider/model info using standard LLM run fields
            try:
                from langsmith import get_current_run_tree
                current_run = get_current_run_tree()
                if current_run:
                    # Update run with LLM-specific properties that LangSmith recognizes
                    # These are the standard fields for LLM runs in LangSmith
                    if hasattr(current_run, 'inputs'):
                        current_run.inputs = current_run.inputs or {}
                        current_run.inputs.update({
                            "model": used_model,
                            "provider": used_provider,
                            "temperature": temperature or settings.llm.default_temperature,
                            "max_tokens": max_tokens or settings.llm.default_max_tokens,
                        })
                    
                    # Set the standard LangSmith LLM run properties
                    current_run.extra = current_run.extra or {}
                    current_run.extra.update({
                        "model": used_model,
                        "provider": used_provider,
                        "generation_time_seconds": generation_time,
                        "invocation_params": {
                            "model": used_model,
                            "temperature": temperature or settings.llm.default_temperature,
                            "max_tokens": max_tokens or settings.llm.default_max_tokens,
                        }
                    })
                    
                    # Update the run name to include provider/model for clarity
                    if used_provider and used_model:
                        current_run.name = f"llm_generate_response [{used_provider}/{used_model}]"
                        
                    # Try to set the run serialized property for model/provider (LangSmith specific)
                    try:
                        if hasattr(current_run, 'serialized'):
                            current_run.serialized = current_run.serialized or {}
                            current_run.serialized.update({
                                "provider": used_provider,
                                "model": used_model
                            })
                    except Exception:
                        pass  # Not all versions support this
                        
            except Exception as e:
                logger.debug(f"Failed to update final LangSmith metadata: {e}")
            
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
        
    def diagnose_provider_config(self, provider: str) -> dict[str, Any]:
        """
        Diagnose provider configuration issues.
        
        Args:
            provider: Provider to diagnose
            
        Returns:
            Configuration diagnosis
        """
        diagnosis = {"provider": provider, "configured": False, "issues": []}
        
        if provider == "openai":
            if settings.llm.openai_api_key:
                diagnosis["configured"] = True
                diagnosis["api_key_length"] = len(settings.llm.openai_api_key)
                diagnosis["api_key_prefix"] = settings.llm.openai_api_key[:8] + "..." if len(settings.llm.openai_api_key) > 8 else "short"
            else:
                diagnosis["issues"].append("LLM_OPENAI_API_KEY not set in environment")
                
        elif provider == "anthropic":
            if settings.llm.anthropic_api_key:
                diagnosis["configured"] = True
                diagnosis["api_key_length"] = len(settings.llm.anthropic_api_key)
                diagnosis["api_key_prefix"] = settings.llm.anthropic_api_key[:8] + "..." if len(settings.llm.anthropic_api_key) > 8 else "short"
            else:
                diagnosis["issues"].append("LLM_ANTHROPIC_API_KEY not set in environment")
                
        elif provider == "ollama":
            if settings.llm.ollama_base_url:
                diagnosis["configured"] = True
                diagnosis["base_url"] = settings.llm.ollama_base_url
            else:
                diagnosis["issues"].append("LLM_OLLAMA_BASE_URL not set in environment")
                
        return diagnosis

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
            # Always test OpenAI if it should be available (even if key is missing)
            if settings.llm.openai_api_key or self.model_type.value == "openai":
                providers_to_test.append("openai")
            if settings.llm.anthropic_api_key or self.model_type.value == "anthropic":
                providers_to_test.append("anthropic")
            if settings.llm.ollama_base_url:
                providers_to_test.append("ollama")
        
        # Default test models for each provider - use current model if available
        test_models = {
            "openai": settings.llm.openai_model if hasattr(settings.llm, 'openai_model') else "gpt-4o-mini",
            "anthropic": settings.llm.anthropic_model if hasattr(settings.llm, 'anthropic_model') else "claude-3-5-haiku-20241022", 
            "ollama": self.default_model if self.model_type.value == "ollama" else "llama3:8b"
        }
                
        for prov in providers_to_test:
            # Add configuration diagnosis
            config_diagnosis = self.diagnose_provider_config(prov)
            
            try:
                # Use current selected model if it matches the provider
                test_model = test_models.get(prov)
                if prov == self.model_type.value:
                    test_model = self.default_model
                    
                response = await self.generate_response(
                    "Test message. Please respond with 'OK'.",
                    provider=prov,
                    model=test_model,
                    max_tokens=10
                )
                results[prov] = {
                    "status": "success",
                    "model": response.model,
                    "response_length": len(response.content),
                    "usage": response.usage,
                    "config": config_diagnosis
                }
            except Exception as e:
                results[prov] = {
                    "status": "error",
                    "error": str(e),
                    "config": config_diagnosis
                }
                
        return results
        
    async def download_ollama_model(self, model_name: str) -> dict[str, Any]:
        """
        Download an Ollama model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Download status and progress info
        """
        if not settings.llm.ollama_base_url:
            return {"status": "error", "error": "Ollama not configured"}
            
        try:
            logger.info(f"Starting download of Ollama model: {model_name}")
            
            async with httpx.AsyncClient(timeout=1800) as client:  # 30 minute timeout for downloads
                response = await client.post(
                    f"{settings.llm.ollama_base_url}/api/pull",
                    json={"name": model_name, "stream": False}
                )
                response.raise_for_status()
                
            result = response.json()
            logger.info(f"Successfully downloaded Ollama model: {model_name}")
            
            return {
                "status": "success",
                "model": model_name,
                "message": f"Model {model_name} downloaded successfully"
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to download Ollama model {model_name}: {e.response.status_code} - {e.response.text}")
            return {
                "status": "error",
                "error": f"Download failed: HTTP {e.response.status_code}"
            }
        except httpx.RequestError as e:
            logger.error(f"Connection error downloading Ollama model {model_name}: {e}")
            return {
                "status": "error", 
                "error": f"Connection error: {e}"
            }
        except Exception as e:
            logger.error(f"Unexpected error downloading Ollama model {model_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

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


def switch_llm_provider(provider: ModelType, model: str | None = None) -> bool:
    """
    Switch LLM provider and optionally model.
    
    Args:
        provider: Provider type (openai, anthropic, ollama)
        model: Specific model to use (optional)
        
    Returns:
        True if switch was successful
    """
    try:
        global _llm_manager
        if _llm_manager is not None:
            _llm_manager.model_type = provider
            if model:
                _llm_manager.default_model = model
        
        # Update settings (note: this doesn't persist to file)
        settings.llm.model_type = provider
        if model:
            settings.llm.default_model = model
            
        logger.info(f"Switched LLM provider to {provider} with model {model or 'default'}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to switch LLM provider: {e}")
        return False


def get_available_models() -> dict[str, list[str]]:
    """Get available models for each provider."""
    return {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ],
        "ollama": [
            "llama3:8b",
            "mistral:7b",
            "qwen3:14b"
        ]
    }


def get_current_llm_config() -> dict[str, Any]:
    """Get current LLM configuration."""
    manager = get_llm_manager()
    return {
        "provider": manager.model_type.value,
        "model": manager.default_model,
        "available_models": get_available_models()
    }


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


def download_ollama_model_sync(model_name: str) -> dict[str, Any]:
    """
    Synchronous wrapper for download_ollama_model.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        Download status
    """
    import asyncio
    
    logger.info(f"[SYNC] Starting download of Ollama model: {model_name}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            manager = get_llm_manager()
            logger.info(f"[SYNC] Created LLM manager, calling async download...")
            result = loop.run_until_complete(manager.download_ollama_model(model_name))
            logger.info(f"[SYNC] Async download completed with result: {result}")
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"[SYNC] Error in sync Ollama model download: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


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