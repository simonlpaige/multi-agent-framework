"""
examples/code_review.py — Automated code review pipeline.

Demonstrates a parallel pipeline where:
  1. Engineer reviews the code for bugs and improvements
  2. Analyst evaluates code complexity and maintainability
  3. CEO aggregates the feedback into a final review summary

Usage:
    python examples/code_review.py
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

from agents.roles import CEOAgent, EngineerAgent, AnalystAgent
from agents.router import MessageRouter
from orchestrator.pipeline import ParallelPipeline
from providers.openai_provider import OpenAIProvider

console = Console()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

# Sample code to review
SAMPLE_CODE = '''
import sqlite3
import os

def get_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def save_users(users):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    for user in users:
        cursor.execute(
            f"INSERT INTO users (name, email, password) VALUES ('{user['name']}', '{user['email']}', '{user['password']}')"
        )
    conn.commit()
    conn.close()

def process_file(filename):
    data = open(filename).read()
    lines = data.split("\\n")
    results = []
    for i in range(len(lines)):
        if lines[i] != "":
            results.append(lines[i].strip())
    return results

class UserCache:
    cache = {}

    def get(self, key):
        return self.cache[key]

    def set(self, key, value):
        self.cache[key] = value

API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"
'''


async def run_code_review(code: str) -> None:
    console.print(Rule("[bold cyan]Automated Code Review Pipeline"))

    # --- Initialize ---
    provider = OpenAIProvider()
    router = MessageRouter()

    engineer = EngineerAgent(provider=provider, router=router)
    analyst = AnalystAgent(provider=provider, router=router)
    ceo = CEOAgent(provider=provider, router=router, name="ReviewLead")

    for agent in [engineer, analyst, ceo]:
        router.register(agent)

    console.print("[dim]Review team:[/dim]", router.list_agents())
    console.print(Panel(code.strip(), title="[yellow]Code Under Review", border_style="yellow"))

    # --- Run parallel review ---
    console.print(Rule("[blue]Running Parallel Reviews"))

    # Give each reviewer a focused task
    engineer.inject_context(
        "You are reviewing the following code. Focus on: SQL injection, "
        "resource leaks, error handling, security vulnerabilities, and bugs.\n\n"
        f"```python\n{code}\n```"
    )
    analyst.inject_context(
        "You are reviewing the following code. Focus on: code quality, "
        "maintainability, design patterns, performance, and best practices.\n\n"
        f"```python\n{code}\n```"
    )

    pipeline = ParallelPipeline(
        agents=[engineer, analyst],
        aggregator=ceo,
    )

    review_task = (
        f"Review this Python code thoroughly. Identify all issues, "
        f"rate severity (critical/major/minor), and suggest fixes.\n\n"
        f"```python\n{code}\n```"
    )

    with console.status("[blue]Engineers and analysts reviewing code..."):
        result = await pipeline.run(review_task)

    # --- Display individual reviews ---
    for stage in result.stages:
        if stage.agent_name != "ReviewLead":
            color = "yellow" if stage.agent_name == "Engineer" else "blue"
            console.print(
                Panel(
                    stage.output,
                    title=f"[{color}]{stage.agent_name} Review ({stage.duration_seconds:.1f}s)",
                    border_style=color,
                )
            )

    # --- Display aggregated review ---
    console.print(Rule("[green]Final Review Summary"))
    console.print(
        Panel(
            result.final_output,
            title="[green]Aggregated Review — ReviewLead",
            border_style="green",
        )
    )

    # --- Pipeline summary ---
    console.print(f"\n[dim]{result.summary()}[/dim]")
    console.print(Rule("[bold cyan]Done"))


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_CODE
    asyncio.run(run_code_review(code))
