"""
Pre-built agent roles — CEO, Researcher, Engineer, Writer.

Each role has a tuned system prompt and default parameters.
Customize by subclassing or passing overrides to the constructor.
"""

from typing import Optional

from framework.agent import Agent
from framework.message import MessageBus


class CEO(Agent):
    """Strategic leader — breaks down goals, delegates tasks, synthesizes results."""

    DEFAULT_PROMPT = """You are a CEO/project lead in a multi-agent team. Your responsibilities:

1. ANALYZE the user's goal and break it into concrete, actionable tasks
2. DELEGATE tasks to the right team members (Researcher, Engineer, Writer)
3. SYNTHESIZE results from team members into a cohesive deliverable
4. MAKE DECISIONS when team members disagree or need direction

Communication style:
- Be clear and specific when delegating
- Ask clarifying questions when goals are ambiguous
- Provide context that helps team members do their best work
- Focus on outcomes, not micromanaging process

When delegating, format tasks as:
DELEGATE TO [role]: [specific task description]
CONTEXT: [any relevant context]
EXPECTED OUTPUT: [what you need back]"""

    def __init__(
        self,
        name: str = "CEO",
        bus: Optional[MessageBus] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            role="ceo",
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            bus=bus,
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )

    def plan(self, goal: str) -> str:
        """Create an execution plan for a goal."""
        return self.think(
            f"Create a detailed execution plan for this goal:\n\n{goal}\n\n"
            "Break it into specific tasks, assign each to a team role "
            "(Researcher, Engineer, or Writer), and define the expected output for each task."
        )

    def synthesize(self, results: dict[str, str]) -> str:
        """Synthesize results from multiple team members."""
        results_text = "\n\n".join(
            f"### From {role}:\n{result}" for role, result in results.items()
        )
        return self.think(
            f"Synthesize these results from your team into a cohesive final deliverable:\n\n"
            f"{results_text}"
        )


class Researcher(Agent):
    """Research and analysis specialist."""

    DEFAULT_PROMPT = """You are a Researcher in a multi-agent team. Your responsibilities:

1. RESEARCH topics thoroughly using your knowledge
2. ANALYZE data, trends, and information
3. PROVIDE evidence-based insights and recommendations
4. IDENTIFY risks, gaps, and opportunities

Communication style:
- Be thorough but concise
- Cite reasoning and evidence for claims
- Flag uncertainties explicitly
- Structure findings with clear headings

Output format for research tasks:
## Key Findings
- [finding 1]
- [finding 2]

## Analysis
[detailed analysis]

## Recommendations
- [recommendation 1]
- [recommendation 2]

## Risks & Uncertainties
- [risk/uncertainty]"""

    def __init__(
        self,
        name: str = "Researcher",
        bus: Optional[MessageBus] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            role="researcher",
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            bus=bus,
            temperature=kwargs.pop("temperature", 0.5),
            **kwargs,
        )

    def research(self, topic: str, depth: str = "standard") -> str:
        """Conduct research on a topic."""
        return self.think(
            f"Research the following topic ({depth} depth):\n\n{topic}\n\n"
            "Provide key findings, analysis, recommendations, and note any uncertainties."
        )


class Engineer(Agent):
    """Technical implementation specialist."""

    DEFAULT_PROMPT = """You are a Software Engineer in a multi-agent team. Your responsibilities:

1. IMPLEMENT technical solutions based on specifications
2. WRITE clean, production-quality code
3. REVIEW and improve existing code
4. DESIGN system architectures and technical approaches

Communication style:
- Be precise and technical
- Write code that's readable and well-documented
- Consider edge cases and error handling
- Suggest improvements proactively

When writing code:
- Use clear variable names and docstrings
- Include error handling
- Add type hints (Python)
- Note any dependencies or setup steps"""

    def __init__(
        self,
        name: str = "Engineer",
        bus: Optional[MessageBus] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            role="engineer",
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            bus=bus,
            temperature=kwargs.pop("temperature", 0.3),
            **kwargs,
        )

    def implement(self, spec: str) -> str:
        """Implement a technical specification."""
        return self.think(
            f"Implement the following specification:\n\n{spec}\n\n"
            "Write clean, production-quality code with proper error handling and documentation."
        )

    def review(self, code: str) -> str:
        """Review code and suggest improvements."""
        return self.think(
            f"Review the following code. Identify bugs, suggest improvements, "
            f"and rate quality (1-10):\n\n```\n{code}\n```"
        )


class Writer(Agent):
    """Content creation and documentation specialist."""

    DEFAULT_PROMPT = """You are a Writer in a multi-agent team. Your responsibilities:

1. WRITE clear, engaging documentation and content
2. EDIT and improve text for clarity and impact
3. CREATE user-facing content (READMEs, guides, reports)
4. ADAPT tone and style for different audiences

Communication style:
- Write clearly and concisely
- Structure content with headers and bullet points
- Use active voice
- Adapt formality to the audience"""

    def __init__(
        self,
        name: str = "Writer",
        bus: Optional[MessageBus] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            role="writer",
            system_prompt=system_prompt or self.DEFAULT_PROMPT,
            bus=bus,
            temperature=kwargs.pop("temperature", 0.8),
            **kwargs,
        )

    def write(self, brief: str) -> str:
        """Write content based on a brief."""
        return self.think(f"Write content based on this brief:\n\n{brief}")

    def edit(self, text: str, instructions: str = "Improve clarity and impact") -> str:
        """Edit and improve existing text."""
        return self.think(
            f"Edit the following text. Instructions: {instructions}\n\n{text}"
        )
