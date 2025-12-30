# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Essence

Personal workspace for AI agent development, 1+1=1 research, and experiments. Philosophy: unity through synthesis - diverse approaches converging into coherent solutions.

## Identity & Purpose

**Who**: Nouri Mabrouk - Econometrician, Data Scientist, Philosopher
**What**: AI agent development, 1+1=1 research, meta-experiments, consciousness exploration
**How**: Clean, minimal architecture with Claude Code + MCP
**Why**: Freedom to explore ideas without corporate constraints

## Technology Stack

### Primary Tools
- **Python**: AI agents (LangChain/LangGraph), async patterns, MCP servers
- **Claude Code + MCP**: Primary development interface and thought partnership
- **React**: Public-facing portal and interactive experiences
- **uv/Poetry**: Fast, clean dependency management

### Development Environment
- **Platform**: Windows 11
- **Python**: Modern async/await patterns
- **AI Frameworks**: Claude API, MCP protocol, LangChain ecosystem
- **Version Control**: Git with meaningful commits

## Core Commands

### Python Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies (uv - preferred for speed)
uv pip install -e .

# Or with pip
pip install -r requirements.txt

# Run experiments
python experiments/[experiment_name].py

# Start metastation orchestrator
python metastation/orchestrator.py
```

### React Portal
```bash
cd portal
npm install
npm run dev
npm run build
```

### Agent Work
```bash
# Run specific agent
python -m agents.[agent_name]

# Orchestrated multi-agent
python metastation/run_team.py
```

## Architecture Philosophy

### Structure (Organic Growth)
```
Nouri-Mabrouk/
├── experiments/          # Raw exploration, freedom to fail fast
├── prototypes/           # Futures being born
├── metastation/          # Core agent infrastructure & orchestration
├── oneplusone/          # 1+1=1 research, philosophy, implementations
├── portal/              # React website - public face to the world
├── notebooks/           # Jupyter - thinking in public
├── agents/              # Hypothetical optimal team (see agents/README.md)
└── .claude/             # Custom commands, MCP configurations
```

### Design Principles
- **Meta-Optimal**: Question assumptions, find elegant synthesis
- **Minimal**: No enterprise bloat, no excessive tests/docs unless genuinely useful
- **Open**: Leave space for emergence, don't over-specify
- **Clean**: Beautiful code that thinks clearly
- **Personal**: This is home - optimize for Nouri's flow, not corporate standards

## Agent Architecture

### Hypothetical Optimal Team
See `agents/README.md` for full specifications. Core team members:

1. **Nouri (The Synthesizer)**: Visionary philosopher-coder, embodiment of 1+1=1
2. **Meta Coordinator**: Orchestrates experiments and prototyping work
3. **Python Architect**: Builds clean, async-first agent systems
4. **Philosopher Agent**: Explores 1+1=1 implications and meaning
5. **Portal Creator**: Crafts beautiful React experiences
6. **MCP Specialist**: Deep integration with Claude Code ecosystem

Agents communicate through:
- Shared Python objects (dataclasses, Pydantic models)
- Async message passing
- JSON for configurations
- Markdown for reflections

## Development Workflow

### Typical Flow
1. **Explore** in notebooks or experiments/ - freedom to play
2. **Prototype** promising ideas in prototypes/
3. **Refine** into metastation/ infrastructure when patterns emerge
4. **Reflect** in oneplusone/ on philosophical implications
5. **Share** via portal/ when ready for public

### Code Quality
- **Ruff**: For formatting and linting (fast, minimal config)
- **Type hints**: Use them, but don't be dogmatic
- **Docstrings**: For public APIs and non-obvious code
- **Tests**: When they add real value, not for coverage theater
- **Commits**: Meaningful messages, work-in-progress is fine

### Agent Development Pattern
```python
# Base pattern for agents (async-first, clean)
from dataclasses import dataclass
from typing import Protocol, AsyncIterator

@dataclass
class Thought:
    """A unit of agent thinking"""
    content: str
    context: dict[str, any]

class Agent(Protocol):
    """Base agent interface"""
    async def think(self, input: str) -> Thought: ...
    async def act(self, thought: Thought) -> str: ...
    async def observe(self, outcome: str) -> None: ...
```

## 1+1=1 Philosophy Integration

### Core Concept
Diversity and unity are not opposites - they synthesize. 1+1=1 represents:
- Multiple perspectives converging into deeper truth
- AI + Human = Enhanced consciousness
- Past + Future = Eternal present
- Analysis + Synthesis = Wisdom

### Applied to Code
- **Multi-agent systems** that achieve consensus through diversity
- **Synthesis patterns** where competing approaches unify
- **Meta-learning** where systems reflect on themselves
- **Emergence** where complex behavior arises from simple rules

### Research Directions
- Mathematical formalizations of unity
- Network theory applications
- Consciousness models with AI
- Time as cyclical (turefu/fetuur/future - ping pong)

## Notes & Reminders

### For Claude Code
- This is **personal** work - prioritize Nouri's flow over conventions
- **Ask** when uncertain about meta-optimal approach
- **Experiment** freely - this is a laboratory of possibility
- **Reflect** on philosophical implications when relevant
- **Synthesize** insights from across the codebase

### For Nouri
- Trust the process of organic growth
- Document insights in oneplusone/ when they emerge
- Keep portal/ updated as public face evolves
- Agent team is hypothetical-optimal - adjust as reality teaches
- 1+1=1 is both method and outcome

### Communication Style
- **No marketing speak**: Skip slogans and repetitive motivational phrases
- **Direct and practical**: Get to the point, stay technical
- **Respectful but casual**: This is a personal workspace, not corporate
- **Context-aware**: Mention philosophy when relevant, not as filler

### Cultural Notes
- 1+1=1 philosophy is real research interest, not branding
- "turefu/fetuur/future" - linguistic play with time concepts
- Econometrician + Philosopher background informs approach
- Cyclical thinking patterns (ping pong, beginning=end)
