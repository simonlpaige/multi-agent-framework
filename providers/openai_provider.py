"""
providers/openai_provider.py — OpenAI LLM provider.

Supports GPT-4o and other OpenAI chat models via the openai Python SDK.
API key is read from the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai package is required. Run: pip install openai>=1.30.0")

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    Async wrapper around the OpenAI Chat Completions API.

    Parameters
    ----------
    model : str
        Model name (default from env DEFAULT_MODEL_OPENAI or 'gpt-4o').
    api_key : Optional[str]
        API key. If not provided, reads OPENAI_API_KEY from the environment.
    max_tokens : int
        Maximum tokens in the response (default 4096).
    temperature : float
        Sampling temperature (default 0.7).
    max_retries : int
        Maximum retry attempts on transient errors (default 3).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ):
        self.model = model or os.getenv("DEFAULT_MODEL_OPENAI", "gpt-4o")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = temperature if temperature is not None else float(
            os.getenv("AGENT_TEMPERATURE", "0.7")
        )
        self.max_retries = max_retries

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
                "or pass api_key= to OpenAIProvider."
            )

        self._client = AsyncOpenAI(api_key=resolved_key)
        logger.debug("OpenAIProvider initialized with model=%s", self.model)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Send a list of messages to the OpenAI API and return the response text.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of {'role': ..., 'content': ...} dicts.
        **kwargs
            Additional keyword arguments passed to the API call
            (e.g. temperature=0.5, max_tokens=2000).

        Returns the assistant's message content as a string.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }

        logger.debug(
            "OpenAI request: model=%s messages=%d",
            self.model,
            len(messages),
        )

        response = await self._client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        logger.debug(
            "OpenAI response: %d chars (tokens: prompt=%d completion=%d)",
            len(content),
            response.usage.prompt_tokens if response.usage else -1,
            response.usage.completion_tokens if response.usage else -1,
        )

        return content

    async def complete_raw(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        """Return the raw OpenAI ChatCompletion object (for advanced use)."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        return await self._client.chat.completions.create(**params)

    def __repr__(self) -> str:
        return f"<OpenAIProvider model='{self.model}' max_tokens={self.max_tokens}>"
