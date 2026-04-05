"""
providers/anthropic_provider.py — Anthropic (Claude) LLM provider.

Supports Claude 3.5 and other Anthropic models via the anthropic Python SDK.
API key is read from the ANTHROPIC_API_KEY environment variable.

Note: Anthropic uses a separate `system` parameter rather than including
system messages in the messages list, so this provider handles that
conversion automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    raise ImportError("anthropic package is required. Run: pip install anthropic>=0.28.0")

logger = logging.getLogger(__name__)


def _split_messages(
    messages: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Split a message list into (system_prompt, user/assistant messages).

    Anthropic requires the system prompt as a separate top-level parameter.
    Multiple system messages are joined with newlines.
    """
    system_parts = []
    chat_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, chat_messages


class AnthropicProvider:
    """
    Async wrapper around the Anthropic Messages API.

    Parameters
    ----------
    model : str
        Model name (default from env DEFAULT_MODEL_ANTHROPIC or 'claude-3-5-sonnet-20241022').
    api_key : Optional[str]
        API key. If not provided, reads ANTHROPIC_API_KEY from the environment.
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
        self.model = model or os.getenv(
            "DEFAULT_MODEL_ANTHROPIC", "claude-3-5-sonnet-20241022"
        )
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = temperature if temperature is not None else float(
            os.getenv("AGENT_TEMPERATURE", "0.7")
        )
        self.max_retries = max_retries

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key= to AnthropicProvider."
            )

        self._client = AsyncAnthropic(api_key=resolved_key)
        logger.debug("AnthropicProvider initialized with model=%s", self.model)

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
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
        Send a list of messages to the Anthropic API and return the response text.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of {'role': ..., 'content': ...} dicts.
            System messages are automatically extracted.
        **kwargs
            Additional keyword arguments passed to the API call.

        Returns the assistant's message content as a string.
        """
        system_prompt, chat_messages = _split_messages(messages)

        # Anthropic requires at least one user message
        if not chat_messages:
            raise ValueError("No user/assistant messages found in message list.")

        # Ensure the conversation starts with a user message
        if chat_messages[0]["role"] != "user":
            raise ValueError(
                "Anthropic API requires the first message to have role='user'."
            )

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Temperature: Anthropic accepts 0.0–1.0 range
        if "temperature" not in kwargs:
            params["temperature"] = min(1.0, max(0.0, self.temperature))

        logger.debug(
            "Anthropic request: model=%s messages=%d system=%d chars",
            self.model,
            len(chat_messages),
            len(system_prompt),
        )

        response = await self._client.messages.create(**params)

        # Extract text from the response content blocks
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        logger.debug(
            "Anthropic response: %d chars (tokens: input=%d output=%d)",
            len(content),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return content

    async def complete_raw(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> anthropic.types.Message:
        """Return the raw Anthropic Message object (for advanced use)."""
        system_prompt, chat_messages = _split_messages(messages)
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
            **kwargs,
        }
        if system_prompt:
            params["system"] = system_prompt
        if "temperature" not in kwargs:
            params["temperature"] = min(1.0, max(0.0, self.temperature))
        return await self._client.messages.create(**params)

    def __repr__(self) -> str:
        return f"<AnthropicProvider model='{self.model}' max_tokens={self.max_tokens}>"
