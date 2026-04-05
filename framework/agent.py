"""
Base Agent class — the building block of multi-agent systems.
"""

import logging
from typing import Optional

from framework.llm import get_completion
from framework.message import Message, MessageBus

logger = logging.getLogger(__name__)


class Agent:
    """An autonomous agent with a role, system prompt, and ability to
    communicate with other agents via a shared message bus.

    Subclass this to create specialized roles (CEO, Researcher, etc.)
    or instantiate directly with a custom system prompt.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        bus: Optional[MessageBus] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.bus = bus or MessageBus()
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._conversation: list[dict] = []

    def think(self, prompt: str) -> str:
        """Send a prompt to the LLM and get a response.

        The agent's system prompt and conversation history are included
        for context continuity.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._conversation)
        messages.append({"role": "user", "content": prompt})

        response = get_completion(
            messages=messages,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
        )

        # Track conversation for context
        self._conversation.append({"role": "user", "content": prompt})
        self._conversation.append({"role": "assistant", "content": response})

        # Keep context window manageable
        if len(self._conversation) > 20:
            self._conversation = self._conversation[-14:]

        logger.info("[%s] thought about: %s...", self.name, prompt[:60])
        return response

    def delegate(self, recipient: str, task: str, context: str = "") -> Message:
        """Delegate a task to another agent via the message bus."""
        content = task
        if context:
            content = f"{task}\n\nContext:\n{context}"

        msg = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            msg_type="task",
        )
        self.bus.send(msg)
        logger.info("[%s] delegated to [%s]: %s", self.name, recipient, task[:60])
        return msg

    def respond(self, to_message: Message, content: str) -> Message:
        """Send a response to a received message."""
        msg = Message(
            sender=self.name,
            recipient=to_message.sender,
            content=content,
            msg_type="response",
            parent_id=to_message.msg_id,
        )
        self.bus.send(msg)
        return msg

    def process_inbox(self) -> list[str]:
        """Process all pending task messages and return responses."""
        tasks = self.bus.get_messages(self.name, msg_type="task")
        responses = []
        for task in tasks:
            logger.info("[%s] processing task from [%s]", self.name, task.sender)
            response_text = self.think(
                f"Task from {task.sender} ({task.msg_type}):\n\n{task.content}"
            )
            self.respond(task, response_text)
            responses.append(response_text)
        return responses

    def reset_context(self) -> None:
        """Clear conversation history."""
        self._conversation = []

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"
