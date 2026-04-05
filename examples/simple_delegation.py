#!/usr/bin/env python3
"""
Example: Simple two-agent delegation.

A CEO delegates a research task to a Researcher, gets the result,
and produces a summary. Minimal example showing the core pattern.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/simple_delegation.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from framework import CEO, Researcher, MessageBus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    bus = MessageBus()

    ceo = CEO(name="Lead", bus=bus)
    researcher = Researcher(name="Analyst", bus=bus)

    # CEO delegates a task
    ceo.delegate("Analyst", "Research the current state of AI code generation tools. "
                 "Compare GitHub Copilot, Cursor, and open-source alternatives. "
                 "Focus on pricing, features, and developer satisfaction.")

    # Researcher processes the task
    responses = researcher.process_inbox()

    # CEO reads the research and synthesizes
    summary = ceo.think(
        f"Your researcher provided this analysis:\n\n{responses[0]}\n\n"
        "Write a 3-sentence executive summary with a recommendation."
    )

    print("\n" + "=" * 60)
    print("RESEARCH RESULT")
    print("=" * 60)
    print(responses[0])
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
