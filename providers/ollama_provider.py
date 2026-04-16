"""
providers/ollama_provider.py — Ollama local model provider.

Routes requests to a locally-running Ollama instance using its
OpenAI-compatible /v1/chat/completions endpoint. No API key required.

Usage::

    from providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="gemma2:27b")
    # or any model pulled via `ollama pull <model>`

Environment variables:
    OLLAMA_BASE_URL   Base URL for Ollama (default: http://localhost:11434)
    DEFAULT_MODEL_OLLAMA  Default model name (default: llama3.2)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai package is required. Run: pip install openai>=1.30.0")


class OllamaProvider:
    """
    Async provider for locally-running Ollama models.

    Uses Ollama's OpenAI-compatible API endpoint, so no separate SDK is needed —
    the standard ``openai`` library handles the transport.

    Parameters
    ----------
    model : str
        Model name as known to Ollama (e.g. ``llama3.2``, ``gemma2:27b``,
        ``mistral``, ``phi4``). Defaults to env var DEFAULT_MODEL_OLLAMA
        or ``llama3.2``.
    base_url : str
        Ollama server URL. Defaults to env var OLLAMA_BASE_URL or
        ``http://localhost:11434``.
    max_tokens : int
        Maximum completion tokens (default 4096).
    temperature : float
        Sampling temperature (default 0.7).
    timeout : float
        Request timeout in seconds. Local models can be slow for large
        prompts; default is 120.0.

    Example::

        import asyncio
        from providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(model="gemma2:27b", temperature=0.3)

        async def main():
            reply = await provider.complete([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarise the CAP theorem in two sentences."},
            ])
            print(reply)

        asyncio.run(main())
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.getenv("DEFAULT_MODEL_OLLAMA", "llama3.2")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = temperature if temperature is not None else float(
            os.getenv("AGENT_TEMPERATURE", "0.7")
        )
        self.timeout = timeout

        # Ollama exposes an OpenAI-compatible endpoint; no real API key needed.
        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",  # required by the SDK, but ignored by Ollama
            timeout=self.timeout,
        )

        logger.debug(
            "OllamaProvider initialized: model=%s base_url=%s",
            self.model,
            self.base_url,
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Send messages to the local Ollama instance and return the reply text.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Conversation history in OpenAI message format.
        **kwargs
            Forwarded to the underlying API call (e.g. ``temperature=0.2``).

        Returns the assistant's message content as a string.

        Raises
        ------
        httpx.ConnectError
            If Ollama is not running at the configured base_url.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }

        logger.debug(
            "Ollama request: model=%s messages=%d",
            self.model,
            len(messages),
        )

        response = await self._client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        logger.debug("Ollama response: %d chars", len(content))
        return content

    async def list_models(self) -> List[str]:
        """
        Return the names of models currently available in this Ollama instance.

        Makes a lightweight GET request to /api/tags — does not use the
        OpenAI-compat layer so it works even before any completions are run.

        Returns a list of model name strings (e.g. ['llama3.2', 'gemma2:27b']).
        """
        import httpx

        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        return [m["name"] for m in data.get("models", [])]

    def __repr__(self) -> str:
        return (
            f"<OllamaProvider model='{self.model}' "
            f"base_url='{self.base_url}' max_tokens={self.max_tokens}>"
        )
