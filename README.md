# multi-agent-framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97757)](https://anthropic.com)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma%204-blue)](https://ollama.com)

A lightweight Python framework for multi-agent orchestration with LLMs. Define agents with roles, wire them up with a message bus, and let them collaborate on complex tasks.

**Used in production** to power [RivalDrop](https://rivaldrop.com) — a competitive intelligence SaaS where CEO, Marketing, Engineering, and Scout agents coordinate autonomously.

## Architecture

```
Team
├── CEO Agent         → plans, delegates, synthesizes
├── Researcher Agent  → analyzes, provides evidence
├── Engineer Agent    → implements, reviews code
├── Writer Agent      → creates docs, content
└── Custom Agents     → SecurityAuditor, Scout, MarketingMgr, etc.
    │
    └── MessageBus    → inter-agent communication (typed, prioritized, threaded)
         │
         └── LLM Provider (OpenAI / Anthropic / Ollama)
```

## Features

- 🤖 **Role-based agents** — CEO, Researcher, Engineer, Writer with tuned system prompts
- 💬 **Message passing** — typed messages with priority, threading, and history
- 🏗️ **Team orchestration** — automatic planning, delegation, and synthesis
- 🔌 **Multi-provider** — OpenAI, Anthropic, and **Ollama (local models)** with unified interface
- 🧩 **Extensible** — subclass `Agent` to create custom roles
- 📝 **Conversation memory** — agents maintain context across interactions
- 🔄 **Heartbeat system** — agents can run on schedules with autonomous check-ins

## Quick Start

```bash
git clone https://github.com/simonlpaige/multi-agent-framework.git
cd multi-agent-framework
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API key (or use Ollama for fully local)

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

### Local-Only (Ollama)

```python
from framework import Team, CEO, Researcher

# Point to local Ollama — no API keys needed
team = Team("Local Team", provider="ollama", model="gemma4:26b")
team.add(CEO(name="Lead"))
team.add(Researcher(name="Analyst"))

results = team.run("Analyze our Q1 sales data and recommend pricing changes")
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
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `gemma4:26b` | Local model name |

## Project Structure

```
multi-agent-framework/
├── framework/
│   ├── __init__.py     # Public API
│   ├── agent.py        # Base Agent class
│   ├── roles.py        # CEO, Researcher, Engineer, Writer
│   ├── team.py         # Team orchestration
│   ├── message.py      # MessageBus + Message dataclass
│   └── llm.py          # OpenAI/Anthropic/Ollama provider abstraction
├── examples/
│   ├── startup_team.py       # Full team collaboration
│   ├── simple_delegation.py  # Two-agent delegation
│   └── local_team.py         # Ollama-only example
├── .env.example
├── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE)
