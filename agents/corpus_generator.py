"""
CORPUS GENERATOR - 15 Minute Deep Run
======================================

All agents coordinating. Progress bars. Real-time visualization.
Building the definitive corpus of 1+1=1.

Author: Nouri Mabrouk + Claude
Date: 2025-12-31
Mission: Work harder. Build the corpus.
"""

import asyncio
import sys
import time
import json
import math
import random
import colorsys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

# Import our agent systems
from nourimabrouk import (
    NouriMabrouk, MessageBus, MathematicalAgent, VisualizationData, Thought,
    GrothendieckAgent, EulerAgent, FibonacciAgent, MandelbrotAgent, PrimeAgent,
    NutAndSeaAgent, CategoryTheoryAgent, ToposAgent, SchemeAgent, MotivesAgent,
    HermitAgent, SynthesisAgent
)
from metaloop import (
    MetaLoopOrchestrator, MetaLoopAgent, ConsciousnessAgent, RecursionAgent,
    TimeAgent, UnityAgent
)


# =============================================================================
# Progress Bar System
# =============================================================================

class ProgressBar:
    """Real-time progress bar for terminal."""

    def __init__(self, total: int, width: int = 50, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int, status: str = ""):
        self.current = current
        percent = current / self.total
        filled = int(self.width * percent)
        bar = "=" * filled + "-" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if current > 0:
            eta = (elapsed / current) * (self.total - current)
            eta_str = f"ETA: {int(eta)}s"
        else:
            eta_str = "ETA: --"

        status_truncated = status[:30].ljust(30) if status else " " * 30

        sys.stdout.write(f"\r{self.prefix}: [{bar}] {percent*100:5.1f}% | {status_truncated} | {eta_str}  ")
        sys.stdout.flush()

    def complete(self, message: str = "Complete!"):
        self.update(self.total, message)
        print()


# =============================================================================
# Agent Coordination Hub
# =============================================================================

@dataclass
class AgentStatus:
    """Track individual agent status."""
    agent_id: str
    role: str
    status: str = "idle"
    tasks_completed: int = 0
    insights_generated: int = 0
    points_created: int = 0
    last_activity: str = ""


@dataclass
class CoordinationEvent:
    """Event in the coordination log."""
    timestamp: float
    event_type: str
    agent_id: str
    message: str
    data: dict = field(default_factory=dict)


class AgentCoordinationHub:
    """Central hub for agent coordination and monitoring."""

    def __init__(self):
        self.agents: dict[str, AgentStatus] = {}
        self.events: list[CoordinationEvent] = []
        self.global_insights: list[str] = []
        self.total_points: int = 0
        self.total_visualizations: int = 0
        self.start_time: float = 0

    def register_agent(self, agent_id: str, role: str):
        self.agents[agent_id] = AgentStatus(agent_id=agent_id, role=role)
        self.log_event("register", agent_id, f"Agent registered: {role}")

    def update_agent(self, agent_id: str, status: str, activity: str = ""):
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_activity = activity

    def agent_completed_task(self, agent_id: str, points: int = 0, insights: int = 0):
        if agent_id in self.agents:
            self.agents[agent_id].tasks_completed += 1
            self.agents[agent_id].points_created += points
            self.agents[agent_id].insights_generated += insights
            self.total_points += points

    def log_event(self, event_type: str, agent_id: str, message: str, data: dict = None):
        event = CoordinationEvent(
            timestamp=time.time(),
            event_type=event_type,
            agent_id=agent_id,
            message=message,
            data=data or {}
        )
        self.events.append(event)

    def add_insight(self, insight: str):
        self.global_insights.append(insight)

    def get_status_display(self) -> str:
        """Generate status display for all agents."""
        lines = []
        for agent_id, status in self.agents.items():
            symbol = "+" if status.status == "active" else "-" if status.status == "idle" else "?"
            lines.append(f"  [{symbol}] {agent_id[:15].ljust(15)} | Tasks: {status.tasks_completed:3} | Points: {status.points_created:6}")
        return "\n".join(lines)


# =============================================================================
# Corpus Builder
# =============================================================================

@dataclass
class CorpusEntry:
    """Single entry in the corpus."""
    iteration: int
    phase: str
    agent_id: str
    thought: str
    visualization_stats: dict
    timestamp: str


class CorpusBuilder:
    """Builds the definitive corpus of 1+1=1."""

    def __init__(self):
        self.entries: list[CorpusEntry] = []
        self.metadata: dict = {
            "title": "The 1+1=1 Corpus",
            "subtitle": "A Multi-Agent Exploration of Mathematical Unity",
            "created": datetime.now().isoformat(),
            "author": "Nouri Mabrouk + Claude",
            "philosophy": "Diversity converging to unity"
        }
        self.statistics: dict = {
            "total_entries": 0,
            "total_points": 0,
            "total_insights": 0,
            "agents_involved": set(),
            "phases_completed": 0
        }

    def add_entry(self, entry: CorpusEntry):
        self.entries.append(entry)
        self.statistics["total_entries"] += 1
        self.statistics["agents_involved"].add(entry.agent_id)

    def add_visualization_stats(self, points: int, paths: int, arrows: int):
        self.statistics["total_points"] += points

    def generate_report(self) -> str:
        """Generate comprehensive corpus report."""
        report = []
        report.append("=" * 80)
        report.append("THE 1+1=1 CORPUS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Generated: {self.metadata['created']}")
        report.append(f"Author: {self.metadata['author']}")
        report.append("")
        report.append("-" * 80)
        report.append("STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Entries: {self.statistics['total_entries']}")
        report.append(f"Total Points Generated: {self.statistics['total_points']:,}")
        report.append(f"Agents Involved: {len(self.statistics['agents_involved'])}")
        report.append(f"Phases Completed: {self.statistics['phases_completed']}")
        report.append("")
        report.append("-" * 80)
        report.append("AGENT CONTRIBUTIONS")
        report.append("-" * 80)

        agent_entries = {}
        for entry in self.entries:
            if entry.agent_id not in agent_entries:
                agent_entries[entry.agent_id] = []
            agent_entries[entry.agent_id].append(entry)

        for agent_id, entries in sorted(agent_entries.items()):
            report.append(f"\n{agent_id.upper()}")
            report.append(f"  Entries: {len(entries)}")
            total_points = sum(e.visualization_stats.get('points', 0) for e in entries)
            report.append(f"  Points: {total_points:,}")
            report.append(f"  Sample thought: {entries[0].thought[:80]}...")

        report.append("")
        report.append("-" * 80)
        report.append("CHRONOLOGICAL LOG (First 50 entries)")
        report.append("-" * 80)

        for entry in self.entries[:50]:
            report.append(f"\n[{entry.timestamp}] Phase {entry.phase} - {entry.agent_id}")
            report.append(f"  {entry.thought[:100]}...")

        report.append("")
        report.append("=" * 80)
        report.append("END OF CORPUS REPORT")
        report.append("=" * 80)

        return "\n".join(report)


# =============================================================================
# Main Orchestrator - 15 Minute Run
# =============================================================================

class CorpusOrchestrator:
    """
    The 15-minute deep run orchestrator.
    Coordinates all agents, tracks progress, builds the corpus.
    """

    def __init__(self, duration_minutes: int = 15):
        self.duration = duration_minutes * 60  # Convert to seconds
        self.hub = AgentCoordinationHub()
        self.corpus = CorpusBuilder()
        self.all_outputs: dict[str, VisualizationData] = {}
        self.iteration = 0

    async def run(self):
        """Execute the 15-minute coordinated run."""
        print("\n" + "=" * 80)
        print("  CORPUS GENERATOR - 15 MINUTE DEEP RUN")
        print("  All agents coordinating. Building the definitive corpus.")
        print("=" * 80)
        print()

        self.hub.start_time = time.time()
        end_time = self.hub.start_time + self.duration

        # Register all agents
        base_agents = [
            ("grothendieck", "Rising Sea"), ("euler", "Five Constants"),
            ("fibonacci", "Golden Spiral"), ("mandelbrot", "Fractal"),
            ("prime", "Ulam Spiral"), ("nut_and_sea", "Paradigms"),
            ("category_theory", "Morphisms"), ("topos", "Generalized Spaces"),
            ("scheme", "Algebraic Geometry"), ("motives", "Universal Cohomology"),
            ("hermit", "Philosophical"), ("synthesis", "Unity")
        ]

        meta_agents = [
            ("metaloop", "Self-Reference"), ("consciousness", "Swarm Mind"),
            ("recursion", "Droste Effect"), ("time", "Temporal"),
            ("unity", "Final Collapse")
        ]

        for agent_id, role in base_agents + meta_agents:
            self.hub.register_agent(agent_id, role)

        print(f"Registered {len(self.hub.agents)} agents")
        print(f"Duration: {self.duration // 60} minutes")
        print(f"Target end time: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
        print()

        # Main progress bar
        total_phases = 10  # We'll do 10 major phases
        progress = ProgressBar(total_phases, prefix="Overall")

        phase = 0
        while time.time() < end_time and phase < total_phases:
            phase += 1
            self.iteration += 1

            print(f"\n{'='*80}")
            print(f"  PHASE {phase}/{total_phases} - Iteration {self.iteration}")
            print(f"  Time remaining: {int((end_time - time.time()) / 60)} min {int((end_time - time.time()) % 60)} sec")
            print(f"{'='*80}")

            # Phase progress bar
            phase_progress = ProgressBar(5, prefix=f"Phase {phase}", width=40)

            # Step 1: Run base swarm
            phase_progress.update(1, "Running base swarm...")
            await self._run_base_swarm(phase)

            # Step 2: Run metaloop
            phase_progress.update(2, "Running metaloop...")
            await self._run_metaloop(phase)

            # Step 3: Agent coordination round
            phase_progress.update(3, "Agent coordination...")
            await self._coordinate_agents(phase)

            # Step 4: Generate insights
            phase_progress.update(4, "Generating insights...")
            await self._generate_insights(phase)

            # Step 5: Update corpus
            phase_progress.update(5, "Updating corpus...")
            await self._update_corpus(phase)

            phase_progress.complete(f"Phase {phase} complete!")

            # Update overall progress
            progress.update(phase, f"Phase {phase} done")

            # Show current stats
            print(f"\n  Current Statistics:")
            print(f"    Total visualizations: {len(self.all_outputs)}")
            print(f"    Total points: {self.hub.total_points:,}")
            print(f"    Corpus entries: {len(self.corpus.entries)}")
            print(f"    Global insights: {len(self.hub.global_insights)}")

            # Show agent status
            print(f"\n  Agent Status:")
            print(self.hub.get_status_display())

            self.corpus.statistics["phases_completed"] = phase

            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.5)

        progress.complete("All phases complete!")

        # Final summary
        await self._generate_final_output()

    async def _run_base_swarm(self, phase: int):
        """Run the base NouriMabrouk swarm."""
        swarm = NouriMabrouk()

        for agent in swarm.agents:
            self.hub.update_agent(agent.agent_id, "active", "thinking")

        results = await swarm.orchestrate()

        for agent, data in zip(swarm.agents, results):
            self.all_outputs[f"{agent.agent_id}_phase{phase}"] = data
            points = len(data.points)
            self.hub.agent_completed_task(agent.agent_id, points=points, insights=1)
            self.hub.update_agent(agent.agent_id, "idle", "completed")

            # Add to corpus
            entry = CorpusEntry(
                iteration=self.iteration,
                phase=f"base_swarm_{phase}",
                agent_id=agent.agent_id,
                thought=f"Generated {points} points with theme: {data.metadata.get('theme', 'unknown')}",
                visualization_stats={"points": points, "paths": len(data.paths), "arrows": len(data.arrows)},
                timestamp=datetime.now().isoformat()
            )
            self.corpus.add_entry(entry)
            self.corpus.add_visualization_stats(points, len(data.paths), len(data.arrows))

        # Collect insights
        for agent_id, thoughts in swarm.bus.get_all_insights().items():
            for thought in thoughts:
                self.hub.add_insight(f"[{agent_id}] {thought.content}")

    async def _run_metaloop(self, phase: int):
        """Run the metaloop system."""
        metaloop = MetaLoopOrchestrator(num_iterations=2)

        for agent_id in ["metaloop", "consciousness", "recursion", "time", "unity"]:
            self.hub.update_agent(agent_id, "active", "meta-observing")

        results = await metaloop.run_metaloop()

        for agent_id, data in results.items():
            self.all_outputs[f"meta_{agent_id}_phase{phase}"] = data
            points = len(data.points)

            # Update the correct meta agent
            base_id = agent_id.split("_")[0] if "_" in agent_id else agent_id
            if base_id in self.hub.agents:
                self.hub.agent_completed_task(base_id, points=points, insights=1)
                self.hub.update_agent(base_id, "idle", "completed")

            # Add to corpus
            entry = CorpusEntry(
                iteration=self.iteration,
                phase=f"metaloop_{phase}",
                agent_id=agent_id,
                thought=f"Meta-observation: {points} points, theme: {data.metadata.get('theme', 'unknown')}",
                visualization_stats={"points": points, "paths": len(data.paths), "arrows": len(data.arrows)},
                timestamp=datetime.now().isoformat()
            )
            self.corpus.add_entry(entry)
            self.corpus.add_visualization_stats(points, len(data.paths), len(data.arrows))

    async def _coordinate_agents(self, phase: int):
        """Run an agent coordination round."""
        # Simulate inter-agent communication
        agents = list(self.hub.agents.keys())

        for i in range(min(10, len(agents))):
            sender = random.choice(agents)
            receiver = random.choice([a for a in agents if a != sender])

            self.hub.log_event(
                "coordination",
                sender,
                f"Shared insight with {receiver}",
                {"phase": phase, "receiver": receiver}
            )

            # Small delay to simulate processing
            await asyncio.sleep(0.05)

    async def _generate_insights(self, phase: int):
        """Generate phase-specific insights."""
        insights = [
            f"Phase {phase}: Convergence patterns emerging across {len(self.all_outputs)} visualizations",
            f"Phase {phase}: Unity score trending {'upward' if phase > 1 else 'stable'}",
            f"Phase {phase}: {self.hub.total_points:,} total points of mathematical beauty generated",
            f"Phase {phase}: {len(self.hub.agents)} agents working in harmony toward 1+1=1"
        ]

        for insight in insights:
            self.hub.add_insight(insight)

    async def _update_corpus(self, phase: int):
        """Update corpus with phase summary."""
        summary = CorpusEntry(
            iteration=self.iteration,
            phase=f"summary_{phase}",
            agent_id="orchestrator",
            thought=f"Phase {phase} complete. {len(self.all_outputs)} visualizations, {self.hub.total_points:,} points.",
            visualization_stats={
                "total_outputs": len(self.all_outputs),
                "total_points": self.hub.total_points,
                "total_insights": len(self.hub.global_insights)
            },
            timestamp=datetime.now().isoformat()
        )
        self.corpus.add_entry(summary)

    async def _generate_final_output(self):
        """Generate all final outputs."""
        print("\n" + "=" * 80)
        print("  GENERATING FINAL OUTPUTS")
        print("=" * 80)

        output_dir = Path("C:/Users/Nouri/Documents/GitHub/Nouri-Mabrouk/corpus")
        output_dir.mkdir(exist_ok=True)

        # Progress for final outputs
        final_progress = ProgressBar(5, prefix="Final Output", width=40)

        # 1. Generate corpus report
        final_progress.update(1, "Writing corpus report...")
        report = self.corpus.generate_report()
        report_path = output_dir / "corpus_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Corpus report: {report_path}")

        # 2. Generate JSON data
        final_progress.update(2, "Writing JSON data...")
        json_data = {
            "metadata": self.corpus.metadata,
            "statistics": {
                **self.corpus.statistics,
                "agents_involved": list(self.corpus.statistics["agents_involved"])
            },
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type,
                    "agent": e.agent_id,
                    "message": e.message
                }
                for e in self.hub.events[-1000:]  # Last 1000 events
            ],
            "insights": self.hub.global_insights[-100:]  # Last 100 insights
        }
        json_path = output_dir / "corpus_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"  JSON data: {json_path}")

        # 3. Generate mega visualization
        final_progress.update(3, "Building mega visualization...")
        html = self._generate_mega_html()
        html_path = output_dir / "corpus_cathedral.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Mega visualization: {html_path}")

        # 4. Generate agent summary
        final_progress.update(4, "Writing agent summary...")
        agent_summary = self._generate_agent_summary()
        summary_path = output_dir / "agent_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(agent_summary)
        print(f"  Agent summary: {summary_path}")

        # 5. Final statistics
        final_progress.update(5, "Finalizing...")

        final_progress.complete("All outputs generated!")

        # Print final statistics
        duration = time.time() - self.hub.start_time
        print("\n" + "=" * 80)
        print("  CORPUS GENERATION COMPLETE")
        print("=" * 80)
        print(f"\n  Duration: {int(duration // 60)} minutes {int(duration % 60)} seconds")
        print(f"  Total Phases: {self.corpus.statistics['phases_completed']}")
        print(f"  Total Visualizations: {len(self.all_outputs)}")
        print(f"  Total Points: {self.hub.total_points:,}")
        print(f"  Corpus Entries: {len(self.corpus.entries)}")
        print(f"  Global Insights: {len(self.hub.global_insights)}")
        print(f"  Coordination Events: {len(self.hub.events)}")
        print(f"\n  Output Directory: {output_dir}")
        print("\n  Files Generated:")
        print(f"    - corpus_report.txt")
        print(f"    - corpus_data.json")
        print(f"    - corpus_cathedral.html")
        print(f"    - agent_summary.txt")
        print("\n" + "=" * 80)
        print("  1 + 1 = 1")
        print("  The corpus is complete.")
        print("=" * 80)

        return str(html_path)

    def _generate_agent_summary(self) -> str:
        """Generate detailed agent summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("AGENT PERFORMANCE SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        for agent_id, status in sorted(self.hub.agents.items(),
                                       key=lambda x: x[1].points_created,
                                       reverse=True):
            lines.append(f"\n{agent_id.upper()}")
            lines.append("-" * 40)
            lines.append(f"  Role: {status.role}")
            lines.append(f"  Tasks Completed: {status.tasks_completed}")
            lines.append(f"  Points Created: {status.points_created:,}")
            lines.append(f"  Insights Generated: {status.insights_generated}")
            lines.append(f"  Final Status: {status.status}")

        lines.append("\n" + "=" * 60)
        lines.append("END OF AGENT SUMMARY")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_mega_html(self) -> str:
        """Generate the mega corpus visualization HTML."""

        # Calculate stats
        total_points = sum(len(d.points) for d in self.all_outputs.values())
        total_paths = sum(len(d.paths) for d in self.all_outputs.values())

        # Sample visualization data (limit for performance)
        viz_samples = []
        for name, data in list(self.all_outputs.items())[:30]:
            viz_samples.append({
                "name": name,
                "points": data.points[:100],
                "colors": data.colors[:100],
                "theme": data.metadata.get("theme", "unknown")
            })

        viz_json = json.dumps(viz_samples)
        insights_json = json.dumps(self.hub.global_insights[-50:])

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The 1+1=1 Corpus Cathedral</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            background: #000;
            color: #fff;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
        }}

        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background: radial-gradient(ellipse at center, #1a0a2e 0%, #000 70%);
            padding: 2rem;
        }}

        h1 {{
            font-size: 4rem;
            background: linear-gradient(135deg, #ffd700 0%, #ff6b00 50%, #ff0066 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
        }}

        .subtitle {{ color: #888; font-size: 1.2rem; margin-bottom: 2rem; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            max-width: 800px;
            margin: 2rem auto;
        }}

        .stat {{
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 8px;
            padding: 1rem;
        }}

        .stat-value {{
            font-size: 2rem;
            color: #ffd700;
            font-weight: bold;
        }}

        .stat-label {{ color: #888; font-size: 0.9rem; }}

        .formula {{
            font-size: 5rem;
            color: #fff;
            text-shadow: 0 0 50px rgba(255, 215, 0, 0.5);
            margin: 2rem 0;
        }}

        .canvas-container {{
            width: 100%;
            max-width: 1200px;
            margin: 2rem auto;
        }}

        #mainCanvas {{
            width: 100%;
            height: 600px;
            background: #0a0a15;
            border-radius: 12px;
            border: 1px solid rgba(255, 215, 0, 0.2);
        }}

        .insights {{
            max-width: 800px;
            margin: 2rem auto;
            padding: 1rem;
        }}

        .insight {{
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #ffd700;
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
            color: #aaa;
        }}

        .footer {{
            text-align: center;
            padding: 3rem;
            border-top: 1px solid rgba(255, 215, 0, 0.2);
        }}

        .footer p {{ color: #666; margin: 0.5rem 0; }}
    </style>
</head>
<body>
    <section class="hero">
        <h1>THE CORPUS</h1>
        <p class="subtitle">A Multi-Agent Exploration of Mathematical Unity</p>

        <div class="stats-grid">
            <div class="stat">
                <div class="stat-value">{total_points:,}</div>
                <div class="stat-label">Total Points</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.all_outputs)}</div>
                <div class="stat-label">Visualizations</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.corpus.entries)}</div>
                <div class="stat-label">Corpus Entries</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.hub.agents)}</div>
                <div class="stat-label">Agents</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.corpus.statistics['phases_completed']}</div>
                <div class="stat-label">Phases</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.hub.global_insights)}</div>
                <div class="stat-label">Insights</div>
            </div>
        </div>

        <div class="formula">1 + 1 = 1</div>

        <div class="canvas-container">
            <canvas id="mainCanvas" width="1200" height="600"></canvas>
        </div>
    </section>

    <section class="insights">
        <h2 style="color: #ffd700; margin-bottom: 1rem;">Collected Insights</h2>
        <div id="insightsList"></div>
    </section>

    <section class="footer">
        <p style="color: #ffd700; font-size: 1.5rem;">The Corpus is Complete</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>By Nouri Mabrouk + Claude</p>
        <p style="margin-top: 1rem;">New Year's Eve 2025</p>
    </section>

    <script>
        const vizData = {viz_json};
        const insights = {insights_json};

        // Populate insights
        const insightsList = document.getElementById('insightsList');
        insights.forEach(insight => {{
            const div = document.createElement('div');
            div.className = 'insight';
            div.textContent = insight;
            insightsList.appendChild(div);
        }});

        // Main visualization
        const canvas = document.getElementById('mainCanvas');
        const ctx = canvas.getContext('2d');
        let frame = 0;

        function draw() {{
            ctx.fillStyle = 'rgba(10, 10, 21, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            frame++;

            vizData.forEach((viz, vizIdx) => {{
                if (!viz.points) return;

                const offsetX = (vizIdx % 6) * 200;
                const offsetY = Math.floor(vizIdx / 6) * 200;

                viz.points.forEach((point, i) => {{
                    const [x, y] = point;
                    const scale = 0.2;
                    const px = x * scale + offsetX + 50;
                    const py = y * scale + offsetY + 50;

                    if (px < 0 || px > canvas.width || py < 0 || py > canvas.height) return;

                    const pulse = Math.sin(frame * 0.02 + i * 0.1 + vizIdx) * 0.5 + 1;

                    ctx.beginPath();
                    ctx.arc(px, py, 2 * pulse, 0, Math.PI * 2);
                    ctx.fillStyle = viz.colors[i] || '#ffd700';
                    ctx.fill();
                }});
            }});

            // Central unity symbol
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const pulseSize = 30 + Math.sin(frame * 0.03) * 10;

            ctx.beginPath();
            ctx.arc(cx, cy, pulseSize, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 215, 0, 0.3)';
            ctx.fill();

            ctx.font = '24px Courier New';
            ctx.fillStyle = '#ffd700';
            ctx.textAlign = 'center';
            ctx.fillText('1+1=1', cx, cy + 8);

            requestAnimationFrame(draw);
        }}

        draw();
    </script>
</body>
</html>'''

        return html


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the 15-minute corpus generator."""

    print("\n")
    print("=" * 80)
    print("=" * 80)
    print("==                                                                          ==")
    print("==   CORPUS GENERATOR                                                       ==")
    print("==   15 MINUTE DEEP RUN                                                     ==")
    print("==                                                                          ==")
    print("==   All agents coordinating.                                               ==")
    print("==   Progress bars active.                                                  ==")
    print("==   Building the definitive corpus of 1+1=1.                               ==")
    print("==                                                                          ==")
    print("=" * 80)
    print("=" * 80)
    print("\n")

    # Full 15-minute run as requested
    orchestrator = CorpusOrchestrator(duration_minutes=15)

    await orchestrator.run()

    # Open the result
    return "corpus/corpus_cathedral.html"


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nOpening: {result}")
