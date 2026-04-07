"""
agents/memory.py — Per-agent and shared memory context.

AgentMemory holds a single agent's conversation history and scratchpad.
SharedMemory acts as a global blackboard accessible by all agents.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


class AgentMemory:
    """Stores an individual agent's message history and key-value scratchpad."""

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_history)
        self._scratchpad: Dict[str, Any] = {}

    # --- History ---

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the history (role = 'user' | 'assistant' | 'system')."""
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the message history."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # --- Scratchpad ---

    def set(self, key: str, value: Any) -> None:
        self._scratchpad[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._scratchpad.get(key, default)

    def delete(self, key: str) -> None:
        self._scratchpad.pop(key, None)

    def all(self) -> Dict[str, Any]:
        return dict(self._scratchpad)

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": list(self._history),
            "scratchpad": self._scratchpad,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_history: int = 50) -> "AgentMemory":
        mem = cls(max_history=max_history)
        for msg in data.get("history", []):
            mem.add_message(msg["role"], msg["content"])
        mem._scratchpad = data.get("scratchpad", {})
        return mem

    # --- Persistence ---

    def save(self, path: str | os.PathLike) -> None:
        """Persist memory to a JSON file. Creates parent directories if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike, max_history: int = 50) -> "AgentMemory":
        """Load memory from a JSON file previously saved with :meth:`save`.

        Args:
            path: Path to the JSON file written by :meth:`save`.
            max_history: Maximum number of messages to keep after loading.

        Returns:
            A new :class:`AgentMemory` instance populated from the file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, max_history=max_history)

    def __repr__(self) -> str:
        return f"<AgentMemory messages={len(self._history)} scratchpad_keys={list(self._scratchpad.keys())}>"


class SharedMemory:
    """
    A thread-safe global blackboard for sharing information between agents.

    Supports typed entries with optional TTL (time-to-live) in seconds.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value. If ttl is given (seconds), the entry expires after that duration."""
        async with self._lock:
            self._store[key] = {
                "value": value,
                "created_at": time.time(),
                "ttl": ttl,
            }

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value. Returns default if missing or expired."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return default
            if entry["ttl"] is not None:
                if time.time() - entry["created_at"] > entry["ttl"]:
                    del self._store[key]
                    return default
            return entry["value"]

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def keys(self) -> List[str]:
        async with self._lock:
            return list(self._store.keys())

    async def all(self) -> Dict[str, Any]:
        """Return all non-expired entries as {key: value}."""
        async with self._lock:
            now = time.time()
            result = {}
            expired = []
            for k, entry in self._store.items():
                if entry["ttl"] is not None and now - entry["created_at"] > entry["ttl"]:
                    expired.append(k)
                else:
                    result[k] = entry["value"]
            for k in expired:
                del self._store[k]
            return result

    async def append_to_list(self, key: str, item: Any) -> None:
        """Convenience: append item to a stored list (creates list if absent)."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None or not isinstance(entry["value"], list):
                self._store[key] = {"value": [item], "created_at": time.time(), "ttl": None}
            else:
                entry["value"].append(item)

    async def snapshot(self) -> str:
        """Return a JSON snapshot of all current (non-expired) entries."""
        data = await self.all()
        return json.dumps(data, default=str, indent=2)

    def __repr__(self) -> str:
        return f"<SharedMemory keys={list(self._store.keys())}>"
