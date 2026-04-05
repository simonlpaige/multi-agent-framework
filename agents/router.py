"""
agents/router.py — Message routing and task delegation between agents.

The MessageRouter maintains a registry of agents and handles async
message delivery. Messages are typed (TASK, QUERY, RESPONSE, BROADCAST)
and can be sent point-to-point or broadcast to all registered agents.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAgent

logger = logging.getLogger(__name__)


class MessageType(Enum):
    TASK = "task"           # Delegate a task to an agent
    QUERY = "query"         # Ask an agent a question
    RESPONSE = "response"   # An agent's reply
    BROADCAST = "broadcast" # Send to all agents
    SYSTEM = "system"       # Internal framework message


@dataclass
class Message:
    """A typed message passed between agents."""

    sender: str
    recipient: str  # agent name or "__all__" for broadcast
    content: str
    message_type: MessageType = MessageType.TASK
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<Message id={self.message_id[:8]} "
            f"type={self.message_type.value} "
            f"from={self.sender} to={self.recipient} "
            f"content={self.content[:60]!r}>"
        )


# Type alias for middleware functions
MiddlewareFn = Callable[[Message], Optional[Message]]


class MessageRouter:
    """
    Central message bus for inter-agent communication.

    Usage
    -----
    router = MessageRouter()
    router.register(my_agent)
    response = await router.send(Message(sender="CEO", recipient="Researcher", content="..."))
    """

    def __init__(self):
        self._agents: Dict[str, "BaseAgent"] = {}
        self._message_log: List[Message] = []
        self._middleware: List[MiddlewareFn] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register(self, agent: "BaseAgent") -> None:
        """Register an agent so it can receive messages."""
        self._agents[agent.name] = agent
        logger.debug("Router: registered agent '%s'", agent.name)

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the registry."""
        self._agents.pop(agent_name, None)

    def get_agent(self, name: str) -> Optional["BaseAgent"]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def add_middleware(self, fn: MiddlewareFn) -> None:
        """
        Add a middleware function that can inspect or transform messages.

        The function receives a Message and must return either:
        - The (optionally modified) Message to continue routing
        - None to drop the message
        """
        self._middleware.append(fn)

    def _apply_middleware(self, message: Message) -> Optional[Message]:
        for fn in self._middleware:
            result = fn(message)
            if result is None:
                logger.debug("Router: message %s dropped by middleware", message.message_id[:8])
                return None
            message = result
        return message

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, message: Message) -> str:
        """
        Route a message to its recipient agent and return the response.

        For BROADCAST messages, sends to all agents and returns a
        newline-joined list of responses.
        """
        # Apply middleware
        message = self._apply_middleware(message)
        if message is None:
            return ""

        async with self._lock:
            self._message_log.append(message)

        if message.message_type == MessageType.BROADCAST or message.recipient == "__all__":
            return await self._broadcast(message)

        return await self._deliver(message)

    async def _deliver(self, message: Message) -> str:
        """Deliver a message to a single agent."""
        agent = self._agents.get(message.recipient)
        if agent is None:
            raise ValueError(
                f"Router: unknown recipient '{message.recipient}'. "
                f"Registered agents: {self.list_agents()}"
            )

        logger.info(
            "Router: [%s] → [%s] (%s) %s",
            message.sender,
            message.recipient,
            message.message_type.value,
            message.content[:60],
        )

        # Dispatch based on message type
        if message.message_type == MessageType.QUERY:
            response = await agent.respond(message.content)
        else:
            # TASK or other types — use act()
            response = await agent.act(message.content)

        # Log the response message
        response_msg = Message(
            sender=message.recipient,
            recipient=message.sender,
            content=response,
            message_type=MessageType.RESPONSE,
            metadata={"reply_to": message.message_id},
        )
        async with self._lock:
            self._message_log.append(response_msg)

        return response

    async def _broadcast(self, message: Message) -> str:
        """Deliver a message to all registered agents concurrently."""
        if not self._agents:
            return ""

        tasks = []
        for agent_name, agent in self._agents.items():
            if agent_name == message.sender:
                continue  # Don't broadcast back to sender
            task_msg = Message(
                sender=message.sender,
                recipient=agent_name,
                content=message.content,
                message_type=MessageType.TASK,
                metadata={"broadcast_id": message.message_id},
            )
            tasks.append(self._deliver(task_msg))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for agent_name, resp in zip(
            [n for n in self._agents if n != message.sender], responses
        ):
            if isinstance(resp, Exception):
                results.append(f"[{agent_name}] ERROR: {resp}")
            else:
                results.append(f"[{agent_name}]: {resp}")

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # Logging & inspection
    # ------------------------------------------------------------------

    def get_message_log(self) -> List[Message]:
        """Return a copy of the full message log."""
        return list(self._message_log)

    def clear_log(self) -> None:
        self._message_log.clear()

    def __repr__(self) -> str:
        return f"<MessageRouter agents={self.list_agents()} messages={len(self._message_log)}>"
