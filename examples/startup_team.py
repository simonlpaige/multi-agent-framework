#!/usr/bin/env python3
"""
Example: Startup team collaborating on a product idea.

Demonstrates multi-agent orchestration with CEO, Researcher,
Engineer, and Writer roles working together.

Usage:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY
    python examples/startup_team.py
"""

import os
import sys
import logging

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from framework import CEO, Researcher, Engineer, Writer, Team

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    # Assemble the team
    team = Team("Startup Team")
    team.add(CEO(name="Alice"))
    team.add(Researcher(name="Bob"))
    team.add(Engineer(name="Carol"))
    team.add(Writer(name="Dave"))

    # Give them a goal
    goal = """
    Design a SaaS product for small businesses to manage customer feedback.
    We need:
    1. Market research on existing solutions
    2. A technical architecture proposal
    3. A landing page copy draft
    
    Budget: bootstrapped. Timeline: MVP in 8 weeks.
    """

    print(f"\n{'='*60}")
    print(f"Team: {team}")
    print(f"Goal: {goal.strip()}")
    print(f"{'='*60}\n")

    # Run the team
    results = team.run(goal, max_rounds=2)

    # Display results
    for agent_name, output in results.items():
        if agent_name == "_synthesis":
            print(f"\n{'='*60}")
            print("FINAL SYNTHESIS")
            print(f"{'='*60}")
        else:
            print(f"\n{'─'*60}")
            print(f"Output from {agent_name}:")
            print(f"{'─'*60}")
        print(output)
        print()


if __name__ == "__main__":
    main()
