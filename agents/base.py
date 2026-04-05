"""
agents/base.py — BaseAgent: the common interface for all agents.

Every agent can:
  think()    — reason about a problem before acting
  act()      — execute a task and produce an output
  delegate() — hand off a subtask to another agent via the router
  respond()  — reply to a message in the context of the conversation
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .memory import AgentMemory

if TYPE_CHECKING:
    from .router import MessageRouter
    from providers.openai_provider import OpenAIProvider
    from providers.anthropic_provider import AnthropicProvider

logger = logging.getLogger(__name__)

ProviderType = Any  # OpenAIProvider | AnthropicProvider


class BaseAgent:
    """
    A self-contained AI agent with memory, a system role prompt, and LLM provider.

    Parameters
    ----------
    name : str
        Human-readable identifier for the agent.
    role : str
        The system prompt that defines the agent's persona and capabilities.
    provider : ProviderType
        An LLM provider instance (OpenAIProvider or AnthropicProvider).
    router : Optional[MessageRouter]
        Message router for inter-agent communication. Required for delegate().
    max_history : int
        Maximum messages to retain in the agent's memory.
    """

    def __init__(
        self,
        name: str,
        role: str,
        provider: ProviderType,
        router: Optional["MessageRouter"] = None,
        max_history: int = 50,
    ):
        self.name = name
        self.role = role
        self.provider = provider
        self.router = router
        self.memory = AgentMemory(max_history=max_history)

        # Seed memory with the system role
        self.memory.add_message("system", role)

        logger.debug("Agent '%s' initialized with provider %s", name, type(provider).__name__)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def think(self, problem: str) -> str:
        """
        Reason about a problem before taking action.

        Sends the problem as a user message prefixed with a chain-of-thought
        instruction, then stores the reasoning in memory.

        Returns the agent's reasoning string.
        """
        prompt = (
            f"Think step-by-step about the following problem and outline your reasoning. "
            f"Do NOT take action yet — just reason.\n\n{problem}"
        )
        self.memory.add_message("user", prompt)
        reasoning = await self.provider.complete(self.memory.get_history())
        self.memory.add_message("assistant", reasoning)
        logger.debug("[%s] think() → %d chars", self.name, len(reasoning))
        return reasoning

    async def act(self, task: str) -> str:
        """
        Execute a task and return the result.

        The task is appended to the existing conversation so the agent retains
        context from prior think() calls.

        Returns the agent's output string.
        """
        self.memory.add_message("user", task)
        result = await self.provider.complete(self.memory.get_history())
        self.memory.add_message("assistant", result)
        logger.debug("[%s] act() → %d chars", self.name, len(result))
        return result

    async def delegate(
        self,
        target_agent_name: str,
        task: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Delegate a subtask to another agent via the router.

        Parameters
        ----------
        target_agent_name : str
            The name of the agent to receive the task.
        task : str
            The task description to send.
        context : Optional[str]
            Additional context to include with the task.

        Returns the delegated agent's response.
        """
        if self.router is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no router attached. "
                "Pass a MessageRouter to delegate tasks."
            )

        full_task = task
        if context:
            full_task = f"Context from {self.name}:\n{context}\n\nTask:\n{task}"

        logger.info("[%s] delegating to '%s': %s", self.name, target_agent_name, task[:80])

        from .router import Message, MessageType

        message = Message(
            sender=self.name,
            recipient=target_agent_name,
            content=full_task,
            message_type=MessageType.TASK,
        )
        response = await self.router.send(message)
        return response

    async def respond(self, user_message: str) -> str:
        """
        Respond to a conversational message, preserving history.

        Use this for multi-turn dialogue where context matters.

        Returns the agent's reply.
        """
        self.memory.add_message("user", user_message)
        reply = await self.provider.complete(self.memory.get_history())
        self.memory.add_message("assistant", reply)
        logger.debug("[%s] respond() → %d chars", self.name, len(reply))
        return reply

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_memory(self, keep_system: bool = True) -> None:
        """Clear conversation history. Optionally re-seed the system prompt."""
        self.memory.clear_history()
        if keep_system:
            self.memory.add_message("system", self.role)

    def inject_context(self, context: str) -> None:
        """Inject a block of context as a system-level message."""
        self.memory.add_message("system", context)

    def get_history(self) -> List[Dict[str, str]]:
        return self.memory.get_history()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' provider={type(self.provider).__name__}>"
