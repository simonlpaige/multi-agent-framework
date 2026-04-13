"""
orchestrator/pipeline.py — Sequential and parallel execution pipelines.

A Pipeline takes a list of agents and runs a prompt through them in order
(sequential) or concurrently (parallel). Each stage can optionally receive
the output of the previous stage as context.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    agent_name: str
    input: str
    output: str
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregated result from a completed pipeline run."""
    stages: List[StageResult] = field(default_factory=list)
    final_output: str = ""
    total_duration_seconds: float = 0.0
    success: bool = True

    def summary(self) -> str:
        lines = [f"Pipeline completed in {self.total_duration_seconds:.2f}s"]
        for s in self.stages:
            status = "✓" if s.success else "✗"
            lines.append(f"  {status} [{s.agent_name}] ({s.duration_seconds:.2f}s)")
        return "\n".join(lines)


# Type alias for optional per-stage transform functions
TransformFn = Callable[[str, str], str]  # (previous_output, stage_prompt) -> new_prompt


class SequentialPipeline:
    """
    Run agents one after another, feeding each agent's output as context
    to the next agent.

    Example
    -------
    pipeline = SequentialPipeline(
        agents=[researcher, analyst, ceo],
        pass_context=True,
    )
    result = await pipeline.run("Analyze the EV market")
    print(result.final_output)
    """

    def __init__(
        self,
        agents: List["BaseAgent"],
        pass_context: bool = True,
        transform: Optional[TransformFn] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ):
        """
        Parameters
        ----------
        agents : List[BaseAgent]
            Agents to run in order.
        pass_context : bool
            If True, each agent receives the previous agent's output as additional context.
        transform : Optional[TransformFn]
            Custom function to build the next stage's prompt from (prev_output, original_task).
        max_retries : int
            Number of retries per stage on failure (0 = no retries). Useful for
            transient API errors that shouldn't kill the whole pipeline.
        retry_delay : float
            Seconds to wait between retries (doubles on each attempt).
        """
        if not agents:
            raise ValueError("SequentialPipeline requires at least one agent.")
        self.agents = agents
        self.pass_context = pass_context
        self.transform = transform
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def run(self, task: str) -> PipelineResult:
        """
        Execute the pipeline.

        Parameters
        ----------
        task : str
            The initial task or prompt to feed to the first agent.

        Returns a PipelineResult with per-stage outputs.
        """
        result = PipelineResult()
        start_total = time.monotonic()
        current_input = task

        for agent in self.agents:
            stage_start = time.monotonic()
            stage_input = current_input

            last_error: Optional[Exception] = None
            for attempt in range(1 + self.max_retries):
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(
                        "Pipeline [%s] retry %d/%d in %.1fs",
                        agent.name, attempt, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    stage_start = time.monotonic()  # reset timer for retry

                try:
                    output = await agent.act(stage_input)
                    duration = time.monotonic() - stage_start
                    stage_result = StageResult(
                        agent_name=agent.name,
                        input=stage_input,
                        output=output,
                        duration_seconds=duration,
                        success=True,
                    )
                    result.stages.append(stage_result)
                    logger.info(
                        "Pipeline [%s] stage complete (%.2fs)",
                        agent.name,
                        duration,
                    )

                    # Build input for the next stage
                    if self.pass_context:
                        if self.transform:
                            current_input = self.transform(output, task)
                        else:
                            current_input = (
                                f"Previous output from {agent.name}:\n{output}\n\n"
                                f"Original task: {task}"
                            )
                    else:
                        current_input = task

                    last_error = None
                    break  # success — move to next stage

                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Pipeline [%s] attempt %d failed: %s",
                        agent.name, attempt + 1, exc,
                    )

            if last_error is not None:
                duration = time.monotonic() - stage_start
                logger.error(
                    "Pipeline [%s] stage failed after %d attempts: %s",
                    agent.name, 1 + self.max_retries, last_error,
                )
                stage_result = StageResult(
                    agent_name=agent.name,
                    input=stage_input,
                    output="",
                    duration_seconds=duration,
                    success=False,
                    error=str(last_error),
                )
                result.stages.append(stage_result)
                result.success = False
                break  # stop pipeline on exhausted retries

        result.total_duration_seconds = time.monotonic() - start_total
        result.final_output = result.stages[-1].output if result.stages else ""
        return result


class ParallelPipeline:
    """
    Run multiple agents concurrently on the same task, then optionally
    merge their outputs with a final aggregator agent.

    Example
    -------
    pipeline = ParallelPipeline(
        agents=[researcher, engineer, analyst],
        aggregator=ceo,
    )
    result = await pipeline.run("Evaluate building an AI startup")
    print(result.final_output)
    """

    def __init__(
        self,
        agents: List["BaseAgent"],
        aggregator: Optional["BaseAgent"] = None,
        max_concurrency: int = 10,
        min_success_ratio: float = 0.5,
    ):
        """
        Parameters
        ----------
        agents : List[BaseAgent]
            Agents to run concurrently.
        aggregator : Optional[BaseAgent]
            If provided, receives all agent outputs and produces a final synthesis.
        max_concurrency : int
            Maximum simultaneous agent calls.
        min_success_ratio : float
            Minimum fraction of agents that must succeed before aggregation
            proceeds. If fewer succeed, the pipeline is marked as failed and
            aggregation is skipped. Default: 0.5 (50%).
        """
        if not agents:
            raise ValueError("ParallelPipeline requires at least one agent.")
        self.agents = agents
        self.aggregator = aggregator
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.min_success_ratio = min_success_ratio

    async def _run_agent(self, agent: "BaseAgent", task: str) -> StageResult:
        start = time.monotonic()
        async with self.semaphore:
            try:
                output = await agent.act(task)
                return StageResult(
                    agent_name=agent.name,
                    input=task,
                    output=output,
                    duration_seconds=time.monotonic() - start,
                    success=True,
                )
            except Exception as exc:
                return StageResult(
                    agent_name=agent.name,
                    input=task,
                    output="",
                    duration_seconds=time.monotonic() - start,
                    success=False,
                    error=str(exc),
                )

    async def run(self, task: str) -> PipelineResult:
        """
        Execute all agents in parallel, then optionally aggregate.

        Returns a PipelineResult. If an aggregator is set, final_output
        is the aggregator's synthesis; otherwise it's all outputs joined.
        """
        result = PipelineResult()
        start_total = time.monotonic()

        # Run all agents concurrently
        stage_tasks = [self._run_agent(agent, task) for agent in self.agents]
        stages = await asyncio.gather(*stage_tasks)
        result.stages = list(stages)
        result.success = all(s.success for s in stages)

        successful = [s for s in stages if s.success]
        failed = [s for s in stages if not s.success]
        success_ratio = len(successful) / len(stages) if stages else 0

        # Check minimum success threshold before aggregating
        if success_ratio < self.min_success_ratio:
            logger.error(
                "ParallelPipeline: only %d/%d agents succeeded (%.0f%%), "
                "below min_success_ratio=%.0f%%. Skipping aggregation.",
                len(successful), len(stages),
                success_ratio * 100, self.min_success_ratio * 100,
            )
            result.success = False
            result.final_output = "\n\n".join(s.output for s in successful)
        elif self.aggregator and successful:
            # Build aggregation prompt that includes both successes and failures
            # so the aggregator knows the synthesis is based on partial data
            outputs_block = "\n\n---\n\n".join(
                f"**{s.agent_name}** (success):\n{s.output}"
                for s in successful
            )
            if failed:
                failures_note = (
                    "\n\n⚠️ **The following agents failed and their output is unavailable:**\n"
                    + "\n".join(
                        f"- **{s.agent_name}**: {s.error or 'unknown error'}"
                        for s in failed
                    )
                    + "\n\nPlease note that this synthesis is based on partial results."
                )
            else:
                failures_note = ""
            agg_prompt = (
                f"You have received the following outputs from your team. "
                f"Synthesize them into a single coherent response that addresses the original task.\n\n"
                f"Original task: {task}\n\n"
                f"Team outputs:\n{outputs_block}"
                f"{failures_note}"
            )
            agg_start = time.monotonic()
            try:
                final = await self.aggregator.act(agg_prompt)
                agg_stage = StageResult(
                    agent_name=self.aggregator.name,
                    input=agg_prompt,
                    output=final,
                    duration_seconds=time.monotonic() - agg_start,
                    success=True,
                )
                result.stages.append(agg_stage)
                result.final_output = final
            except Exception as exc:
                logger.error("ParallelPipeline aggregator failed: %s", exc)
                result.final_output = "\n\n".join(s.output for s in successful)
        else:
            result.final_output = "\n\n".join(s.output for s in successful)

        result.total_duration_seconds = time.monotonic() - start_total
        return result
