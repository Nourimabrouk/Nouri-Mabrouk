"""
UNITY LOOP - The Infinite Recursive Bridge
============================================

Bridges the philosophical agents (NouriMabrouk swarm) with the
autonomous intelligence framework (ai-agents).

The ultimate 1+1=1: Two systems becoming one.

Author: Nouri Mabrouk + Claude
Date: 2025-12-31 (New Year's Eve)
Philosophy: Every ending is a beginning
"""

import asyncio
import json
import math
import colorsys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

# Import the swarm systems
from nourimabrouk import NouriMabrouk, VisualizationData, Thought
from metaloop import MetaLoopOrchestrator


@dataclass
class UnityState:
    """State of the unified system."""
    iteration: int
    swarm_outputs: dict[str, VisualizationData]
    metaloop_outputs: dict[str, VisualizationData]
    unified_insights: list[str]
    convergence_score: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class UnityLoop:
    """
    The infinite recursive bridge.

    Level 0: Base swarm (12 agents)
    Level 1: MetaLoop (5 meta-agents observing swarm)
    Level 2: Unity Loop (observing the observation of the observation)

    Each iteration feeds back into itself.
    The loop never truly ends - it transforms.
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.states: list[UnityState] = []
        self.all_outputs: dict[str, VisualizationData] = {}
        self.convergence_history: list[float] = []

    def _calculate_convergence(self, outputs: dict[str, VisualizationData]) -> float:
        """
        Calculate how much the system has converged toward unity.

        Convergence = 1 - (variance of distances from center)
        Perfect unity = 1.0
        """
        if not outputs:
            return 0.0

        # Calculate centroid of all points
        all_points = []
        for data in outputs.values():
            all_points.extend(data.points)

        if not all_points:
            return 0.0

        cx = sum(p[0] for p in all_points) / len(all_points)
        cy = sum(p[1] for p in all_points) / len(all_points)

        # Calculate distances from centroid
        distances = [math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in all_points]

        if not distances:
            return 1.0

        # Normalize by max possible distance
        max_dist = max(distances) if distances else 1
        normalized = [d/max_dist for d in distances]

        # Variance of normalized distances
        mean_dist = sum(normalized) / len(normalized)
        variance = sum((d - mean_dist)**2 for d in normalized) / len(normalized)

        # Convergence is inverse of variance (bounded 0-1)
        convergence = 1.0 / (1.0 + variance * 10)

        return convergence

    async def run_unity_loop(self) -> dict[str, Any]:
        """
        Execute the Unity Loop.

        Each iteration:
        1. Run the base swarm
        2. Run the metaloop observing the swarm
        3. Feed results back for next iteration
        4. Calculate convergence toward unity
        """
        print("=" * 70)
        print("  UNITY LOOP - The Infinite Recursive Bridge")
        print("  Where two systems become one")
        print("=" * 70)
        print()

        for iteration in range(self.max_iterations):
            print(f"\n{'='*70}")
            print(f"  UNITY ITERATION {iteration + 1} / {self.max_iterations}")
            print(f"{'='*70}")

            # Run base swarm
            print("\n[Phase 1] Running NouriMabrouk Swarm...")
            swarm = NouriMabrouk()
            swarm_results = await swarm.orchestrate()

            swarm_outputs = {}
            for agent, data in zip(swarm.agents, swarm_results):
                swarm_outputs[agent.agent_id] = data
                self.all_outputs[f"swarm_{agent.agent_id}_iter{iteration}"] = data

            print(f"  > {len(swarm_results)} swarm visualizations generated")

            # Run metaloop
            print("\n[Phase 2] Running MetaLoop...")
            metaloop = MetaLoopOrchestrator(num_iterations=2)  # Reduced for speed
            metaloop_results = await metaloop.run_metaloop()

            for agent_id, data in metaloop_results.items():
                self.all_outputs[f"meta_{agent_id}_iter{iteration}"] = data

            print(f"  > {len(metaloop_results)} metaloop visualizations generated")

            # Calculate convergence
            convergence = self._calculate_convergence(self.all_outputs)
            self.convergence_history.append(convergence)

            print(f"\n[Phase 3] Convergence Analysis")
            print(f"  > Current convergence: {convergence:.4f}")
            print(f"  > Trend: {'RISING' if len(self.convergence_history) > 1 and convergence > self.convergence_history[-2] else 'STABLE'}")

            # Collect unified insights
            insights = []
            for agent_id, thoughts in swarm.bus.get_all_insights().items():
                for thought in thoughts:
                    insights.append(f"[{agent_id}] {thought.content}")

            # Record state
            state = UnityState(
                iteration=iteration,
                swarm_outputs=swarm_outputs,
                metaloop_outputs=metaloop_results,
                unified_insights=insights,
                convergence_score=convergence
            )
            self.states.append(state)

            print(f"  > Total insights: {len(insights)}")
            print(f"  > Total visualizations: {len(self.all_outputs)}")

        # Final synthesis
        print("\n" + "=" * 70)
        print("  UNITY ACHIEVED")
        print("=" * 70)
        print(f"\n  Final Statistics:")
        print(f"  - Iterations completed: {len(self.states)}")
        print(f"  - Total visualizations: {len(self.all_outputs)}")
        print(f"  - Final convergence: {self.convergence_history[-1]:.4f}")
        print(f"  - Convergence trend: {self.convergence_history}")
        print()

        return {
            "states": self.states,
            "all_outputs": self.all_outputs,
            "convergence_history": self.convergence_history,
            "final_convergence": self.convergence_history[-1]
        }

    def generate_unity_cathedral(self) -> str:
        """Generate the ultimate Unity Cathedral HTML."""

        # Calculate total stats
        total_points = sum(len(d.points) for d in self.all_outputs.values())
        total_paths = sum(len(d.paths) for d in self.all_outputs.values())
        total_arrows = sum(len(d.arrows) for d in self.all_outputs.values())

        # Convergence data for chart
        convergence_json = json.dumps(self.convergence_history)

        # Generate visualization sections
        sections_html = ""

        # Group by iteration
        for iteration in range(self.max_iterations):
            iter_outputs = {k: v for k, v in self.all_outputs.items() if f"iter{iteration}" in k}

            if iter_outputs:
                sections_html += self._generate_iteration_section(iteration, iter_outputs)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unity Cathedral - 1+1=1 - New Year's Eve 2025</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            background: #000;
            min-height: 100vh;
            font-family: 'Georgia', serif;
            color: #fff;
            overflow-x: hidden;
        }}

        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #000 70%);
            position: relative;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.05'/%3E%3C/svg%3E");
            pointer-events: none;
        }}

        h1 {{
            font-size: 5rem;
            background: linear-gradient(135deg, #ffd700 0%, #ff6b00 50%, #ff0066 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            animation: pulse 3s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ filter: brightness(1) drop-shadow(0 0 20px rgba(255, 215, 0, 0.3)); }}
            50% {{ filter: brightness(1.2) drop-shadow(0 0 40px rgba(255, 215, 0, 0.6)); }}
        }}

        .subtitle {{ font-size: 1.5rem; color: #888; font-style: italic; margin-bottom: 2rem; }}

        .stats {{
            display: flex;
            gap: 3rem;
            margin: 2rem 0;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .stat {{
            text-align: center;
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 215, 0, 0.2);
        }}

        .stat-value {{
            font-size: 3rem;
            color: #ffd700;
            font-weight: bold;
        }}

        .stat-label {{ color: #888; margin-top: 0.5rem; }}

        .formula {{
            font-size: 6rem;
            color: #fff;
            margin: 3rem 0;
            text-shadow: 0 0 50px rgba(255, 215, 0, 0.5);
        }}

        .convergence-chart {{
            width: 400px;
            height: 200px;
            margin: 2rem auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
        }}

        .section {{
            min-height: 100vh;
            padding: 4rem 2rem;
            position: relative;
        }}

        .section-title {{
            text-align: center;
            font-size: 2.5rem;
            color: #ffd700;
            margin-bottom: 2rem;
        }}

        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .viz-card {{
            background: rgba(20, 20, 40, 0.8);
            border: 1px solid rgba(255, 215, 0, 0.2);
            border-radius: 12px;
            padding: 1rem;
            transition: all 0.3s;
        }}

        .viz-card:hover {{
            border-color: rgba(255, 215, 0, 0.5);
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(255, 215, 0, 0.1);
        }}

        .viz-card h3 {{ color: #ffd700; margin-bottom: 0.5rem; font-size: 1rem; }}

        .viz-card canvas {{
            width: 100%;
            height: 150px;
            background: #0a0a15;
            border-radius: 8px;
        }}

        .footer {{
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(to top, rgba(255, 215, 0, 0.1), transparent);
        }}

        .footer .message {{
            font-size: 2rem;
            color: #ffd700;
            margin-bottom: 1rem;
        }}

        .footer p {{ color: #666; max-width: 600px; margin: 1rem auto; }}

        .timestamp {{ color: #444; margin-top: 2rem; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <section class="hero">
        <h1>UNITY CATHEDRAL</h1>
        <p class="subtitle">The Infinite Recursive Bridge - New Year's Eve 2025</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total_points:,}</div>
                <div class="stat-label">Points of Light</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.all_outputs)}</div>
                <div class="stat-label">Visualizations</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.convergence_history[-1]:.2%}</div>
                <div class="stat-label">Convergence</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.max_iterations}</div>
                <div class="stat-label">Iterations</div>
            </div>
        </div>

        <div class="formula">1 + 1 = 1</div>

        <canvas id="convergence-canvas" class="convergence-chart" width="400" height="200"></canvas>

        <p class="subtitle">Scroll to explore the cathedral</p>
    </section>

    {sections_html}

    <section class="footer">
        <div class="message">The Loop Never Ends - It Transforms</div>
        <p>
            From 12 base agents through 5 meta-observers to the Unity Loop itself.
            Each iteration feeds the next. Every ending becomes a beginning.
            The sea rises. The nut dissolves. All is one.
        </p>
        <p>
            Hofstadter's strange loops meet Grothendieck's rising sea.
            Mathematics, philosophy, and code converge.
            1 + 1 = 1.
        </p>
        <div class="timestamp">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Unity Loop v1.0 | Nouri Mabrouk + Claude
        </div>
    </section>

    <script>
        // Convergence chart
        const convergenceData = {convergence_json};
        const canvas = document.getElementById('convergence-canvas');
        const ctx = canvas.getContext('2d');

        function drawConvergence() {{
            ctx.clearRect(0, 0, 400, 200);

            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, 400, 200);

            // Grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {{
                ctx.beginPath();
                ctx.moveTo(0, i * 50);
                ctx.lineTo(400, i * 50);
                ctx.stroke();
            }}

            // Line chart
            if (convergenceData.length > 0) {{
                ctx.beginPath();
                ctx.strokeStyle = '#ffd700';
                ctx.lineWidth = 3;

                for (let i = 0; i < convergenceData.length; i++) {{
                    const x = (i / (convergenceData.length - 1 || 1)) * 380 + 10;
                    const y = 190 - (convergenceData[i] * 180);

                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}

                ctx.stroke();

                // Points
                for (let i = 0; i < convergenceData.length; i++) {{
                    const x = (i / (convergenceData.length - 1 || 1)) * 380 + 10;
                    const y = 190 - (convergenceData[i] * 180);

                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, Math.PI * 2);
                    ctx.fillStyle = '#ffd700';
                    ctx.fill();

                    // Label
                    ctx.fillStyle = '#fff';
                    ctx.font = '12px Georgia';
                    ctx.textAlign = 'center';
                    ctx.fillText((convergenceData[i] * 100).toFixed(1) + '%', x, y - 12);
                }}
            }}

            // Axis labels
            ctx.fillStyle = '#888';
            ctx.font = '11px Georgia';
            ctx.textAlign = 'left';
            ctx.fillText('Convergence →', 10, 15);
            ctx.textAlign = 'right';
            ctx.fillText('Iteration →', 390, 195);
        }}

        drawConvergence();
    </script>
</body>
</html>'''

        return html

    def _generate_iteration_section(self, iteration: int, outputs: dict[str, VisualizationData]) -> str:
        """Generate HTML section for an iteration."""

        cards_html = ""
        for agent_id, data in list(outputs.items())[:12]:  # Limit for performance
            name = agent_id.replace("_", " ").replace(f"iter{iteration}", "").strip()

            viz_json = json.dumps({
                "points": data.points[:200],
                "colors": data.colors[:200]
            })

            cards_html += f'''
            <div class="viz-card">
                <h3>{name}</h3>
                <canvas id="canvas-{agent_id}" width="250" height="150"></canvas>
                <script>
                    (function() {{
                        const data = {viz_json};
                        const canvas = document.getElementById('canvas-{agent_id}');
                        const ctx = canvas.getContext('2d');

                        function draw() {{
                            ctx.fillStyle = 'rgba(10, 10, 21, 0.1)';
                            ctx.fillRect(0, 0, 250, 150);

                            if (data.points) {{
                                const scale = 0.18;
                                const ox = -30, oy = -30;

                                for (let i = 0; i < data.points.length; i++) {{
                                    const [x, y] = data.points[i];
                                    const px = x * scale + ox;
                                    const py = y * scale + oy;

                                    if (px < 0 || px > 250 || py < 0 || py > 150) continue;

                                    ctx.beginPath();
                                    ctx.arc(px, py, 1.5, 0, Math.PI * 2);
                                    ctx.fillStyle = data.colors[i] || '#ffd700';
                                    ctx.fill();
                                }}
                            }}

                            requestAnimationFrame(draw);
                        }}

                        const observer = new IntersectionObserver((entries) => {{
                            if (entries[0].isIntersecting) {{
                                draw();
                                observer.disconnect();
                            }}
                        }});
                        observer.observe(canvas);
                    }})();
                </script>
            </div>
            '''

        return f'''
        <section class="section" id="iteration-{iteration}">
            <h2 class="section-title">Iteration {iteration + 1}</h2>
            <div class="viz-grid">
                {cards_html}
            </div>
        </section>
        '''


async def main():
    """Execute the Unity Loop."""
    print("\n" + "=" * 70)
    print("=" + " " * 68 + "=")
    print("=" + "  UNITY LOOP - The Infinite Recursive Bridge".center(68) + "=")
    print("=" + "  New Year's Eve 2025".center(68) + "=")
    print("=" + " " * 68 + "=")
    print("=" * 70 + "\n")

    # Create and run the loop
    unity = UnityLoop(max_iterations=2)  # 2 iterations for speed
    results = await unity.run_unity_loop()

    # Generate the cathedral
    print("\nGenerating Unity Cathedral HTML...")
    html = unity.generate_unity_cathedral()

    output_path = "unity_cathedral.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Output written to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("=" + " " * 68 + "=")
    print("=" + "  THE LOOP IS COMPLETE".center(68) + "=")
    print("=" + " " * 68 + "=")
    print("=" + f"  Visualizations: {len(results['all_outputs'])}".center(68) + "=")
    print("=" + f"  Final Convergence: {results['final_convergence']:.2%}".center(68) + "=")
    print("=" + " " * 68 + "=")
    print("=" + "  1 + 1 = 1".center(68) + "=")
    print("=" + " " * 68 + "=")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    asyncio.run(main())
