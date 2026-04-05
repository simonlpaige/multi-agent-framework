"""
Team — orchestrates multiple agents working together on a goal.
"""

import logging
from typing import Optional

from framework.agent import Agent
from framework.message import MessageBus

logger = logging.getLogger(__name__)


class Team:
    """A team of agents that collaborate to accomplish goals.

    The Team manages a shared message bus and orchestrates agent
    interactions. Optionally designates a lead agent who plans
    and delegates work.

    Example:
        team = Team("Product Team")
        team.add(CEO(name="Alice"))
        team.add(Researcher(name="Bob"))
        team.add(Engineer(name="Carol"))
        result = team.run("Build a REST API for user management")
    """

    def __init__(self, name: str = "Team", lead: Optional[str] = None):
        self.name = name
        self.bus = MessageBus()
        self.agents: dict[str, Agent] = {}
        self._lead = lead

    def add(self, agent: Agent) -> "Team":
        """Add an agent to the team. Sets the shared message bus."""
        agent.bus = self.bus
        self.agents[agent.name] = agent
        if self._lead is None and agent.role == "ceo":
            self._lead = agent.name
        logger.info("[%s] Added agent: %s (%s)", self.name, agent.name, agent.role)
        return self

    @property
    def lead(self) -> Optional[Agent]:
        """The team lead agent (usually CEO)."""
        return self.agents.get(self._lead) if self._lead else None

    def run(self, goal: str, max_rounds: int = 3) -> dict[str, str]:
        """Execute a goal with the team.

        The lead agent creates a plan and delegates tasks.
        Other agents process their inbox and respond.
        The lead synthesizes results.

        Args:
            goal: The objective to accomplish.
            max_rounds: Maximum delegation/response rounds.

        Returns:
            Dict mapping agent names to their final outputs.
        """
        if not self.lead:
            raise ValueError(
                "Team needs a lead agent. Add a CEO or set lead= in constructor."
            )

        logger.info("[%s] Starting goal: %s", self.name, goal[:80])
        results: dict[str, str] = {}

        # Step 1: Lead creates a plan
        plan = self.lead.think(
            f"You are leading a team of: {', '.join(f'{a.name} ({a.role})' for a in self.agents.values() if a.name != self.lead.name)}.\n\n"
            f"Goal: {goal}\n\n"
            "Create a plan. For each task, delegate to a specific team member."
        )
        results[self.lead.name] = plan
        logger.info("[%s] Plan created by %s", self.name, self.lead.name)

        # Step 2: Parse delegations and send tasks
        for agent_name, agent in self.agents.items():
            if agent_name == self.lead.name:
                continue
            # Ask the lead what this agent should do
            task_prompt = (
                f"Based on your plan, what specific task should {agent_name} ({agent.role}) "
                f"work on? Be specific and actionable. Respond with just the task."
            )
            task = self.lead.think(task_prompt)
            self.lead.delegate(agent_name, task)

        # Step 3: Agents process tasks
        for round_num in range(max_rounds):
            logger.info("[%s] Round %d/%d", self.name, round_num + 1, max_rounds)
            any_work = False
            for agent_name, agent in self.agents.items():
                if agent_name == self.lead.name:
                    continue
                tasks = self.bus.get_messages(agent_name, msg_type="task")
                if tasks:
                    any_work = True
                    responses = agent.process_inbox()
                    if responses:
                        results[agent_name] = responses[-1]

            if not any_work:
                break

        # Step 4: Lead synthesizes
        if len(results) > 1:
            synthesis = self.lead.synthesize(
                {k: v for k, v in results.items() if k != self.lead.name}
            )
            results["_synthesis"] = synthesis

        logger.info("[%s] Complete. %d agents contributed.", self.name, len(results))
        return results

    def chat(self, agent_name: str, message: str) -> str:
        """Send a direct message to a specific agent and get a response."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"No agent named '{agent_name}'. Available: {list(self.agents.keys())}")
        return agent.think(message)

    def __repr__(self) -> str:
        members = ", ".join(f"{a.name}({a.role})" for a in self.agents.values())
        return f"Team({self.name!r}, [{members}])"
