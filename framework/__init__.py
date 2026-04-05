"""Multi-Agent Framework — lightweight agent orchestration for LLMs."""

from framework.agent import Agent
from framework.roles import CEO, Researcher, Engineer, Writer
from framework.team import Team
from framework.message import Message, MessageBus

__all__ = [
    "Agent",
    "CEO",
    "Researcher",
    "Engineer",
    "Writer",
    "Team",
    "Message",
    "MessageBus",
]
