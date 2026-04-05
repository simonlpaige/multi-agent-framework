# multi-agent-framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97757)](https://anthropic.com)

A lightweight Python framework for multi-agent orchestration with LLMs. Define agents with roles, wire them up with a message bus, and let them collaborate on complex tasks.

## Architecture

```
Team
├── CEO Agent         → plans, delegates, synthesizes
├── Researcher Agent  → analyzes, provides evidence
├── Engineer Agent    → implements, reviews code
└── Writer Agent      → creates docs, content
    │
    └── MessageBus    → inter-agent communication
         │
         └── LLM Provider (OpenAI / Anthropic)
```

## Features

- 🤖 **Role-based agents** — CEO, Researcher, Engineer, Writer with tuned system prompts
- 💬 **Message passing** — typed messages with priority, threading, and history
- 🏗️ **Team orchestration** — automatic planning, delegation, and synthesis
- 🔌 **Multi-provider** — OpenAI and Anthropic with a unified interface
- 🧩 **Extensible** — subclass `Agent` to create custom roles
- 📝 **Conversation memory** — agents maintain context across interactions

## Quick Start

```bash
git clone https://github.com/simonlpaige/multi-agent-framework.git
cd multi-agent-framework
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API key

python examples/startup_team.py
```

## Usage

### Basic: Two agents collaborating

```python
from framework import CEO, Researcher, MessageBus

bus = MessageBus()
ceo = CEO(name="Lead", bus=bus)
researcher = Researcher(name="Analyst", bus=bus)

# CEO delegates a task
ceo.delegate("Analyst", "Research the competitive landscape for AI coding tools.")

# Researcher processes and responds
responses = researcher.process_inbox()

# CEO synthesizes the result
summary = ceo.think(f"Based on this research:\n{responses[0]}\nWrite a summary.")
```

### Team: Full orchestration

```python
from framework import CEO, Researcher, Engineer, Writer, Team

team = Team("Product Team")
team.add(CEO(name="Alice"))
team.add(Researcher(name="Bob"))
team.add(Engineer(name="Carol"))
team.add(Writer(name="Dave"))

results = team.run("Design a REST API for user management with docs")

# results["_synthesis"] contains the final combined output
print(results["_synthesis"])
```

### Custom Agent

```python
from framework import Agent

class SecurityAuditor(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Auditor",
            role="security",
            system_prompt="You are a security expert. Review systems for vulnerabilities...",
            temperature=0.2,
            **kwargs,
        )

    def audit(self, target: str) -> str:
        return self.think(f"Audit this system for security issues:\n{target}")
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model to use |

## Project Structure

```
multi-agent-framework/
├── framework/
│   ├── __init__.py     # Public API
│   ├── agent.py        # Base Agent class
│   ├── roles.py        # CEO, Researcher, Engineer, Writer
│   ├── team.py         # Team orchestration
│   ├── message.py      # MessageBus + Message dataclass
│   └── llm.py          # OpenAI/Anthropic provider abstraction
├── examples/
│   ├── startup_team.py       # Full team collaboration
│   └── simple_delegation.py  # Two-agent delegation
├── .env.example
├── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE)
