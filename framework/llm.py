"""
LLM provider abstraction.
Supports OpenAI and Anthropic with a unified interface.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_completion(
    messages: list[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Get a chat completion from the configured LLM provider.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        provider: "openai" or "anthropic". Defaults to LLM_PROVIDER env var.
        model: Model name override.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        The assistant's response text.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return _openai_completion(messages, model, temperature, max_tokens)
    elif provider == "anthropic":
        return _anthropic_completion(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _openai_completion(
    messages: list[dict],
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _anthropic_completion(
    messages: list[dict],
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Anthropic uses a system parameter instead of system messages
    system_msg = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg += m["content"] + "\n"
        else:
            chat_messages.append(m)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_msg.strip() if system_msg else None,
        messages=chat_messages,
    )
    return response.content[0].text
