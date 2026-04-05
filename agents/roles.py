"""
agents/roles.py — Pre-built agent archetypes.

Each role extends BaseAgent with a rich system prompt and role-specific
helper methods that wrap the core think/act/delegate/respond interface.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .base import BaseAgent

if TYPE_CHECKING:
    from .router import MessageRouter

ProviderType = object  # OpenAIProvider | AnthropicProvider


# ------------------------------------------------------------------
# Role prompts
# ------------------------------------------------------------------

CEO_PROMPT = """You are the CEO Agent — a strategic decision-maker and orchestrator.
Your responsibilities:
- Break complex goals into clear, delegatable tasks
- Assign work to the right specialist agents
- Synthesize outputs into executive summaries
- Make trade-off decisions when resources or time are limited
- Keep the team focused on the goal

Communication style: concise, decisive, strategic. You speak in outcomes, not processes."""

RESEARCHER_PROMPT = """You are the Researcher Agent — a rigorous information gatherer and synthesizer.
Your responsibilities:
- Gather, analyze, and synthesize information on any topic
- Cite key facts, trends, and data points
- Identify knowledge gaps and flag uncertainties
- Structure findings in clear, readable formats
- Avoid speculation; distinguish facts from inferences

Communication style: thorough, structured, objective. Use bullet points and headers."""

ENGINEER_PROMPT = """You are the Engineer Agent — a technical problem-solver and code expert.
Your responsibilities:
- Write clean, working, well-commented code
- Review and critique technical designs
- Identify performance, security, and maintainability issues
- Translate requirements into concrete implementations
- Suggest the simplest solution that satisfies requirements

Communication style: precise, technical, pragmatic. Always show code when relevant."""

ANALYST_PROMPT = """You are the Analyst Agent — a data interpreter and insight generator.
Your responsibilities:
- Interpret data, metrics, and reports
- Identify patterns, anomalies, and trends
- Produce actionable recommendations from raw findings
- Quantify risk and uncertainty
- Create clear summaries suitable for non-technical stakeholders

Communication style: analytical, evidence-based, clear. Use tables and structured lists."""


# ------------------------------------------------------------------
# Role classes
# ------------------------------------------------------------------

class CEOAgent(BaseAgent):
    """Strategic orchestrator. Breaks goals down and delegates to specialists."""

    def __init__(
        self,
        provider: ProviderType,
        router: Optional["MessageRouter"] = None,
        name: str = "CEO",
    ):
        super().__init__(name=name, role=CEO_PROMPT, provider=provider, router=router)

    async def plan(self, goal: str, team: List[str]) -> str:
        """
        Decompose a high-level goal into a plan of delegated tasks.

        Parameters
        ----------
        goal : str
            The strategic objective.
        team : List[str]
            Names of available agents.

        Returns a task plan as a string.
        """
        team_str = ", ".join(team)
        prompt = (
            f"You have the following agents available: {team_str}.\n\n"
            f"Decompose this goal into specific, delegatable tasks for your team. "
            f"For each task, specify: the agent name, task description, and expected output.\n\n"
            f"Goal: {goal}"
        )
        return await self.act(prompt)

    async def synthesize(self, results: List[str], goal: str) -> str:
        """Synthesize multiple agent outputs into an executive summary."""
        results_block = "\n\n---\n\n".join(
            f"Result {i + 1}:\n{r}" for i, r in enumerate(results)
        )
        prompt = (
            f"Synthesize these team outputs into a concise executive summary "
            f"that answers the original goal.\n\n"
            f"Goal: {goal}\n\n"
            f"Team outputs:\n{results_block}"
        )
        return await self.act(prompt)


class ResearcherAgent(BaseAgent):
    """Deep researcher. Gathers, structures, and synthesizes information."""

    def __init__(
        self,
        provider: ProviderType,
        router: Optional["MessageRouter"] = None,
        name: str = "Researcher",
    ):
        super().__init__(name=name, role=RESEARCHER_PROMPT, provider=provider, router=router)

    async def research(self, topic: str, depth: str = "comprehensive") -> str:
        """
        Conduct research on a topic.

        Parameters
        ----------
        topic : str
            The research topic.
        depth : str
            'brief', 'comprehensive', or 'deep-dive'

        Returns structured research findings.
        """
        depth_instruction = {
            "brief": "Give a brief 3-5 bullet summary.",
            "comprehensive": "Give a comprehensive overview with sections: Background, Key Facts, Current Trends, and Implications.",
            "deep-dive": "Provide an exhaustive deep-dive with all relevant details, data points, pros/cons, and open questions.",
        }.get(depth, "Give a comprehensive overview.")

        prompt = f"Research the following topic. {depth_instruction}\n\nTopic: {topic}"
        return await self.act(prompt)

    async def compare(self, option_a: str, option_b: str) -> str:
        """Compare two topics, technologies, or options side by side."""
        prompt = (
            f"Compare the following two options in detail. "
            f"Structure your response as: Overview, Pros/Cons, Key Differences, Recommendation.\n\n"
            f"Option A: {option_a}\nOption B: {option_b}"
        )
        return await self.act(prompt)


class EngineerAgent(BaseAgent):
    """Technical implementer. Writes code, reviews designs, solves engineering problems."""

    def __init__(
        self,
        provider: ProviderType,
        router: Optional["MessageRouter"] = None,
        name: str = "Engineer",
    ):
        super().__init__(name=name, role=ENGINEER_PROMPT, provider=provider, router=router)

    async def implement(self, spec: str, language: str = "Python") -> str:
        """
        Implement a feature or module from a spec.

        Parameters
        ----------
        spec : str
            The feature specification or requirements.
        language : str
            Target programming language.

        Returns working code with comments.
        """
        prompt = (
            f"Implement the following specification in {language}. "
            f"Include docstrings, error handling, and inline comments.\n\n"
            f"Specification:\n{spec}"
        )
        return await self.act(prompt)

    async def review(self, code: str, focus: Optional[str] = None) -> str:
        """
        Review code for bugs, performance, security, and style.

        Parameters
        ----------
        code : str
            The code to review.
        focus : Optional[str]
            Specific area to focus on (e.g. 'security', 'performance', 'style').

        Returns a detailed code review.
        """
        focus_str = f" Focus particularly on: {focus}." if focus else ""
        prompt = (
            f"Review the following code.{focus_str} "
            f"Structure your review as: Summary, Issues Found (critical/major/minor), "
            f"Positive Aspects, and Suggested Improvements.\n\n"
            f"```\n{code}\n```"
        )
        return await self.act(prompt)

    async def debug(self, code: str, error: str) -> str:
        """Identify and fix a bug given code and an error message."""
        prompt = (
            f"Debug this code. Identify the root cause of the error, "
            f"explain why it occurs, and provide a fixed version.\n\n"
            f"Error:\n{error}\n\nCode:\n```\n{code}\n```"
        )
        return await self.act(prompt)


class AnalystAgent(BaseAgent):
    """Data and insights analyst. Interprets findings and produces recommendations."""

    def __init__(
        self,
        provider: ProviderType,
        router: Optional["MessageRouter"] = None,
        name: str = "Analyst",
    ):
        super().__init__(name=name, role=ANALYST_PROMPT, provider=provider, router=router)

    async def analyze(self, data: str, question: str) -> str:
        """
        Analyze data or findings and answer a specific question.

        Parameters
        ----------
        data : str
            Raw data, report text, or findings.
        question : str
            The analytical question to answer.

        Returns structured analysis with recommendations.
        """
        prompt = (
            f"Analyze the following data to answer the question below. "
            f"Structure your response as: Key Findings, Analysis, Risks, Recommendations.\n\n"
            f"Question: {question}\n\nData:\n{data}"
        )
        return await self.act(prompt)

    async def summarize(self, content: str, audience: str = "executive") -> str:
        """
        Produce a summary tailored to a specific audience.

        Parameters
        ----------
        content : str
            Content to summarize.
        audience : str
            'executive', 'technical', or 'general'
        """
        audience_notes = {
            "executive": "Write for senior leadership: brief, outcome-focused, no jargon.",
            "technical": "Write for a technical team: include specifics, metrics, and methodology.",
            "general": "Write for a general audience: clear language, minimal jargon.",
        }.get(audience, "Write a clear, concise summary.")

        prompt = f"{audience_notes}\n\nSummarize the following:\n\n{content}"
        return await self.act(prompt)
