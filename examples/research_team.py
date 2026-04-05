"""
examples/research_team.py — Multi-agent research workflow.

Demonstrates a three-agent pipeline:
  1. Researcher   — gathers information on the topic
  2. Analyst      — interprets the research and draws insights
  3. CEO          — synthesizes into an executive summary

Usage:
    python examples/research_team.py
    python examples/research_team.py "quantum computing breakthroughs 2025"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from agents.roles import CEOAgent, ResearcherAgent, AnalystAgent
from agents.router import MessageRouter
from agents.memory import SharedMemory
from orchestrator.pipeline import SequentialPipeline
from providers.openai_provider import OpenAIProvider

console = Console()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))


async def run_research_team(topic: str) -> None:
    console.print(Rule("[bold cyan]Multi-Agent Research Team"))
    console.print(f"\n[bold]Topic:[/bold] {topic}\n")

    # --- Initialize provider ---
    provider = OpenAIProvider()

    # --- Initialize shared memory ---
    shared_memory = SharedMemory()

    # --- Initialize router ---
    router = MessageRouter()

    # --- Create agents ---
    researcher = ResearcherAgent(provider=provider, router=router)
    analyst = AnalystAgent(provider=provider, router=router)
    ceo = CEOAgent(provider=provider, router=router)

    # Register all agents with the router
    for agent in [researcher, analyst, ceo]:
        router.register(agent)

    console.print("[dim]Agents registered:[/dim]", router.list_agents())

    # --- Step 1: Researcher gathers information ---
    console.print(Rule("[yellow]Step 1: Research"))
    with console.status(f"[yellow]{researcher.name} is researching..."):
        research_findings = await researcher.research(topic, depth="comprehensive")

    await shared_memory.set("research_findings", research_findings)
    console.print(
        Panel(research_findings, title=f"[yellow]{researcher.name}", border_style="yellow")
    )

    # --- Step 2: Analyst interprets the findings ---
    console.print(Rule("[blue]Step 2: Analysis"))
    with console.status(f"[blue]{analyst.name} is analyzing..."):
        analysis = await analyst.analyze(
            data=research_findings,
            question=f"What are the most important insights and implications of: {topic}?",
        )

    await shared_memory.set("analysis", analysis)
    console.print(
        Panel(analysis, title=f"[blue]{analyst.name}", border_style="blue")
    )

    # --- Step 3: CEO synthesizes into executive summary ---
    console.print(Rule("[green]Step 3: Executive Summary"))
    with console.status(f"[green]{ceo.name} is synthesizing..."):
        summary = await ceo.synthesize(
            results=[research_findings, analysis],
            goal=f"Provide strategic recommendations on: {topic}",
        )

    await shared_memory.set("executive_summary", summary)
    console.print(
        Panel(summary, title=f"[green]{ceo.name} — Executive Summary", border_style="green")
    )

    # --- Optional: Delegation demo ---
    console.print(Rule("[magenta]Bonus: CEO delegates a follow-up task"))
    with console.status("[magenta]Delegating follow-up research..."):
        follow_up = await ceo.delegate(
            target_agent_name="Researcher",
            task="What are the top 3 companies leading this space and why?",
            context=summary,
        )

    console.print(
        Panel(follow_up, title="[magenta]Follow-up (via delegation)", border_style="magenta")
    )

    # --- Print router log ---
    console.print(Rule("[dim]Message Log"))
    for msg in router.get_message_log():
        console.print(f"  [dim]{msg}[/dim]")

    console.print(Rule("[bold cyan]Done"))


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "the rise of AI agents in enterprise software 2025"
    asyncio.run(run_research_team(topic))
