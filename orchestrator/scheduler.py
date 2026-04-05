"""
orchestrator/scheduler.py — Priority-based async task scheduler.

Tasks are submitted with a priority (lower number = higher priority) and
dispatched to their assigned agents. The scheduler supports:
- Priority queue ordering
- Dependency declaration (task B waits for task A)
- Concurrency limiting
- Per-task timeout
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"   # Dependency failed


@dataclass
class Task:
    """
    A unit of work to be dispatched to an agent.

    Parameters
    ----------
    name : str
        Human-readable task name.
    prompt : str
        The prompt/instruction for the agent.
    agent : BaseAgent
        The agent that will execute this task.
    priority : int
        Scheduling priority. Lower number = higher priority (default 5).
    dependencies : List[str]
        Task IDs that must complete successfully before this task runs.
    timeout : Optional[float]
        Seconds before the task is cancelled (None = no timeout).
    metadata : Dict[str, Any]
        Arbitrary metadata attached to the task.
    """

    name: str
    prompt: str
    agent: "BaseAgent"
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Managed by scheduler
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    result: Optional[str] = field(default=None, init=False)
    error: Optional[str] = field(default=None, init=False)
    started_at: Optional[float] = field(default=None, init=False)
    completed_at: Optional[float] = field(default=None, init=False)

    # For heap comparison (priority, creation order)
    _seq: int = field(default=0, init=False)

    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def __lt__(self, other: "Task") -> bool:
        # Heap orders by (priority, seq) so lower priority number runs first
        return (self.priority, self._seq) < (other.priority, other._seq)

    def __repr__(self) -> str:
        return (
            f"<Task id={self.task_id[:8]} name='{self.name}' "
            f"priority={self.priority} status={self.status.value}>"
        )


class TaskScheduler:
    """
    Async priority task scheduler with dependency resolution.

    Example
    -------
    scheduler = TaskScheduler(max_concurrency=3)

    t1 = scheduler.add_task("Research", "Research EV market trends", researcher)
    t2 = scheduler.add_task("Analyze", "Analyze findings", analyst, dependencies=[t1.task_id])

    results = await scheduler.run_all()
    for task in results:
        print(task.name, task.status, task.result[:100])
    """

    def __init__(self, max_concurrency: int = 5):
        """
        Parameters
        ----------
        max_concurrency : int
            Maximum tasks running simultaneously.
        """
        self.max_concurrency = max_concurrency
        self._tasks: Dict[str, Task] = {}
        self._heap: List[Task] = []
        self._seq_counter = 0
        self._semaphore: Optional[asyncio.Semaphore] = None

    def add_task(
        self,
        name: str,
        prompt: str,
        agent: "BaseAgent",
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Create and enqueue a task.

        Returns the Task object (use task.task_id for dependency refs).
        """
        task = Task(
            name=name,
            prompt=prompt,
            agent=agent,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            metadata=metadata or {},
        )
        task._seq = self._seq_counter
        self._seq_counter += 1
        self._tasks[task.task_id] = task
        heapq.heappush(self._heap, task)
        logger.debug("Scheduled task '%s' (priority=%d, id=%s)", name, priority, task.task_id[:8])
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    async def run_all(self) -> List[Task]:
        """
        Execute all enqueued tasks respecting priority and dependencies.

        Returns the list of all tasks (inspect .status and .result).
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        # We process in waves: ready tasks run concurrently; once done,
        # newly unlocked tasks are dispatched.
        completed_ids: set = set()
        failed_ids: set = set()

        while True:
            ready = self._get_ready_tasks(completed_ids, failed_ids)
            if not ready:
                # Check if anything is still pending (stuck dependencies)
                pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
                if pending:
                    # Mark stuck tasks as skipped
                    for t in pending:
                        t.status = TaskStatus.SKIPPED
                        t.error = "Dependency never resolved"
                        logger.warning("Task '%s' skipped: unresolved dependencies", t.name)
                break

            run_tasks = [self._execute_task(t) for t in ready]
            await asyncio.gather(*run_tasks)

            for t in ready:
                if t.status == TaskStatus.COMPLETED:
                    completed_ids.add(t.task_id)
                else:
                    failed_ids.add(t.task_id)

        return list(self._tasks.values())

    def _get_ready_tasks(
        self, completed_ids: set, failed_ids: set
    ) -> List[Task]:
        """Return all PENDING tasks whose dependencies are satisfied."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            # Check dependencies
            deps = task.dependencies
            if any(dep_id in failed_ids for dep_id in deps):
                task.status = TaskStatus.SKIPPED
                task.error = "A dependency task failed"
                logger.warning("Task '%s' skipped: dependency failed", task.name)
                continue
            if all(dep_id in completed_ids for dep_id in deps):
                ready.append(task)

        # Sort by priority (lower number first)
        ready.sort(key=lambda t: (t.priority, t._seq))
        return ready

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task, respecting the concurrency semaphore."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = time.monotonic()
            logger.info("Running task '%s' on agent '%s'", task.name, task.agent.name)

            try:
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.agent.act(task.prompt),
                        timeout=task.timeout,
                    )
                else:
                    result = await task.agent.act(task.prompt)

                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.monotonic()
                logger.info(
                    "Task '%s' completed in %.2fs",
                    task.name,
                    task.duration() or 0,
                )

            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Timed out after {task.timeout}s"
                task.completed_at = time.monotonic()
                logger.error("Task '%s' timed out", task.name)

            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.completed_at = time.monotonic()
                logger.error("Task '%s' failed: %s", task.name, exc)

    def summary(self) -> str:
        """Return a human-readable summary of all task statuses."""
        lines = ["Task Scheduler Summary", "=" * 40]
        status_icons = {
            TaskStatus.COMPLETED: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "~",
            TaskStatus.PENDING: "○",
            TaskStatus.RUNNING: "◎",
        }
        for task in sorted(self._tasks.values(), key=lambda t: t._seq):
            icon = status_icons.get(task.status, "?")
            dur = f"{task.duration():.2f}s" if task.duration() else "—"
            lines.append(f"  {icon} [{task.priority}] {task.name} ({dur})")
            if task.error:
                lines.append(f"      Error: {task.error}")
        return "\n".join(lines)
