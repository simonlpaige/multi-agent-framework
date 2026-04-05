"""
Message passing system for inter-agent communication.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message passed between agents."""

    sender: str
    recipient: str
    content: str
    msg_type: str = "task"  # task, response, info, delegation
    priority: int = 0       # higher = more important
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    msg_id: str = field(default_factory=lambda: f"msg-{int(time.time()*1000)}")


class MessageBus:
    """Central message bus for agent communication.

    Agents post messages to the bus; recipients poll or get pushed messages.
    Supports filtering by recipient, type, and priority.
    """

    def __init__(self):
        self._messages: list[Message] = []
        self._subscribers: dict[str, list] = {}  # agent_name -> callback list

    def send(self, message: Message) -> None:
        """Post a message to the bus."""
        self._messages.append(message)
        logger.debug(
            "[%s → %s] %s: %s",
            message.sender,
            message.recipient,
            message.msg_type,
            message.content[:80],
        )

        # Notify subscribers
        for callback in self._subscribers.get(message.recipient, []):
            callback(message)

    def subscribe(self, agent_name: str, callback) -> None:
        """Register a callback for messages to a specific agent."""
        self._subscribers.setdefault(agent_name, []).append(callback)

    def get_messages(
        self,
        recipient: str,
        msg_type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> list[Message]:
        """Retrieve messages for a recipient with optional filters."""
        results = [m for m in self._messages if m.recipient == recipient]
        if msg_type:
            results = [m for m in results if m.msg_type == msg_type]
        if since:
            results = [m for m in results if m.timestamp > since]
        return sorted(results, key=lambda m: -m.priority)

    def get_conversation(self, agent_a: str, agent_b: str) -> list[Message]:
        """Get all messages exchanged between two agents, in order."""
        msgs = [
            m
            for m in self._messages
            if (m.sender in (agent_a, agent_b) and m.recipient in (agent_a, agent_b))
        ]
        return sorted(msgs, key=lambda m: m.timestamp)

    @property
    def history(self) -> list[Message]:
        """All messages in chronological order."""
        return sorted(self._messages, key=lambda m: m.timestamp)
