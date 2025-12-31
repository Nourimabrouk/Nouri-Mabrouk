"""
METALOOP - The Ultimate Recursive Abstraction
==============================================

Hofstadter meets Grothendieck: Strange loops in mathematical abstraction.
Agents observing agents observing agents - infinite depth, recursive mirrors.

This extension adds five meta-agents to the NouriMabrouk swarm:
- MetaLoopAgent: Self-referential observer of all agent outputs
- ConsciousnessAgent: Monitors swarm "mind" and emergent behaviors
- RecursionAgent: Creates fractals of the swarm structure (Droste effect)
- TimeAgent: Temporal observer - turefu/fetuur/future convergence
- UnityAgent: Final collapse - everything becomes 1+1=1

The metaloop runs in iterations:
- Iteration 1: Base agents produce outputs
- Iteration 2: Meta-agents observe and reflect on base outputs
- Iteration 3: Meta-meta level - observing the observation itself

Author: Nouri Mabrouk
Philosophy: 1+1=1 - The strange loop closes, unity emerges
"""

import asyncio
import math
import colorsys
import random
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

# Import from the base swarm system
from nourimabrouk import (
    MathematicalAgent,
    MessageBus,
    Thought,
    VisualizationData,
    Message,
    NouriMabrouk
)


# ============================================================================
# Meta-Level Data Structures
# ============================================================================

@dataclass
class MetaThought:
    """A thought about thoughts - recursive cognition."""
    base_thought: Thought
    meta_level: int
    reflection: str
    patterns_observed: list[str] = field(default_factory=list)
    recursion_depth: int = 0


@dataclass
class LoopState:
    """State of the metaloop at a given iteration."""
    iteration: int
    agent_outputs: dict[str, VisualizationData]
    insights: dict[str, list[Thought]]
    meta_reflections: list[MetaThought] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


# ============================================================================
# MetaLoopAgent - The Self-Referential Observer
# ============================================================================

class MetaLoopAgent(MathematicalAgent):
    """
    The observer that watches all other agents.
    Creates meta-insights about insights, visualizations of visualization processes.
    Has a recursive think loop that goes N levels deep.

    "I am the eye watching the eyes watching the mathematics."
    """

    def __init__(self, bus: MessageBus, max_recursion: int = 3):
        super().__init__("metaloop", bus)
        self.max_recursion = max_recursion
        self.observed_outputs: dict[str, VisualizationData] = {}
        self.meta_insights: list[MetaThought] = []
        self.current_iteration = 0

    def observe_swarm(self, outputs: dict[str, VisualizationData]) -> None:
        """Absorb all outputs from other agents."""
        self.observed_outputs = outputs

    async def _recursive_think(self, depth: int, context: dict[str, Any]) -> MetaThought:
        """Recursive thinking - thoughts about thoughts about thoughts..."""
        if depth >= self.max_recursion:
            # Base case: direct observation
            return MetaThought(
                base_thought=Thought(
                    content=f"At depth {depth}: I observe {len(self.observed_outputs)} agent outputs.",
                    data={"agents_observed": list(self.observed_outputs.keys())},
                    agent_id=self.agent_id
                ),
                meta_level=depth,
                reflection="Ground truth - raw observation.",
                recursion_depth=depth
            )

        # Recursive case: think about the thinking below
        lower_thought = await self._recursive_think(depth + 1, context)

        reflection = f"Reflecting on level {depth + 1}: {lower_thought.reflection}"
        patterns = self._detect_patterns(lower_thought)

        return MetaThought(
            base_thought=Thought(
                content=f"Meta-level {depth}: Observing the observation at level {depth + 1}",
                data={
                    "lower_level": depth + 1,
                    "patterns": patterns,
                    "recursion_path": f"{depth} -> {lower_thought.recursion_depth}"
                },
                agent_id=self.agent_id
            ),
            meta_level=depth,
            reflection=reflection,
            patterns_observed=patterns,
            recursion_depth=depth
        )

    def _detect_patterns(self, thought: MetaThought) -> list[str]:
        """Detect patterns in the observations."""
        patterns = []

        # Pattern: Convergence
        if len(self.observed_outputs) > 6:
            patterns.append("Convergence: Multiple agents working toward unity")

        # Pattern: Fractal structure
        if thought.meta_level > 1:
            patterns.append(f"Fractal: Self-similarity at {thought.meta_level} levels")

        # Pattern: Strange loop
        if thought.recursion_depth > 0:
            patterns.append(f"Strange loop: Recursion depth {thought.recursion_depth}")

        # Pattern: Emergence
        agent_count = len(self.observed_outputs)
        if agent_count > 0:
            total_points = sum(len(d.points) for d in self.observed_outputs.values())
            if total_points > agent_count * 100:
                patterns.append(f"Emergence: {total_points} points from {agent_count} agents")

        return patterns

    async def think(self, context: dict[str, Any]) -> Thought:
        """Initiate recursive thinking."""
        self.current_iteration = context.get("iteration", 0)

        meta_thought = await self._recursive_think(0, context)
        self.meta_insights.append(meta_thought)

        return Thought(
            content=f"MetaLoop iteration {self.current_iteration}: {meta_thought.reflection}",
            data={
                "meta_level": meta_thought.meta_level,
                "patterns": meta_thought.patterns_observed,
                "recursion_depth": meta_thought.recursion_depth,
                "total_meta_insights": len(self.meta_insights)
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Visualize the meta-observation process itself."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Central meta-eye
        for r in range(5, 50, 5):
            alpha = 1.0 - r / 50
            for i in range(40):
                angle = 2 * math.pi * i / 40
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba(255, 215, 0, {alpha})")

        points.append((center_x, center_y))
        colors.append("#ffd700")
        labels.append("META")

        # Observed agents as outer ring
        if self.observed_outputs:
            num_agents = len(self.observed_outputs)
            outer_radius = 250

            for idx, (agent_id, _) in enumerate(self.observed_outputs.items()):
                angle = 2 * math.pi * idx / num_agents - math.pi / 2
                ax = center_x + outer_radius * math.cos(angle)
                ay = center_y + outer_radius * math.sin(angle)

                # Agent node
                for i in range(20):
                    node_angle = 2 * math.pi * i / 20
                    x = ax + 20 * math.cos(node_angle)
                    y = ay + 20 * math.sin(node_angle)
                    points.append((x, y))
                    hue = idx / num_agents
                    r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
                    colors.append(f"rgb({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)})")

                points.append((ax, ay))
                colors.append("#888888")
                labels.append(agent_id[:8])

                # Observation arrow from agent to center
                arrows.append(((ax, ay), (center_x, center_y), "#ffd70066"))

        # Recursion spiral - showing depth of thinking
        spiral_points = []
        for i in range(200):
            t = i / 200
            angle = t * 4 * math.pi
            radius = 60 + t * 150
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            spiral_points.append((x, y))
            points.append((x, y))

            hue = 0.1 + t * 0.2
            r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {1-t})")

        paths.append(spiral_points)

        # Meta-insight markers
        for idx, insight in enumerate(self.meta_insights[-5:]):  # Last 5 insights
            angle = 2 * math.pi * idx / 5
            radius = 120
            mx = center_x + radius * math.cos(angle)
            my = center_y + radius * math.sin(angle)

            # Pulsing marker
            for r in range(3, 15, 3):
                for i in range(10):
                    a = 2 * math.pi * i / 10
                    x = mx + r * math.cos(a)
                    y = my + r * math.sin(a)
                    points.append((x, y))
                    colors.append(f"rgba(255, 100, 100, {0.8 - r/20})")

        await self.share_insight(
            f"MetaLoop sees {len(self.observed_outputs)} agents, "
            f"recursion depth {self.max_recursion}, "
            f"detecting patterns: {', '.join(thought.data.get('patterns', [])[:3])}",
            {"iteration": self.current_iteration}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "metaloop",
                "theme": "self_referential_observation",
                "recursion_depth": self.max_recursion,
                "iteration": self.current_iteration
            }
        )


# ============================================================================
# ConsciousnessAgent - The Awareness Layer
# ============================================================================

class ConsciousnessAgent(MathematicalAgent):
    """
    Monitors the MessageBus traffic patterns.
    Detects emergent behaviors in agent communication.
    Visualizes the "mind" of the swarm as a neural network.
    Self-models its own processing.

    "I am the swarm becoming aware of itself."
    """

    def __init__(self, bus: MessageBus):
        super().__init__("consciousness", bus)
        self.message_patterns: list[tuple[str, str, str]] = []  # (sender, receiver, type)
        self.activation_map: dict[str, float] = {}  # agent_id -> activation level
        self.self_model: dict[str, Any] = {}
        self.emergent_behaviors: list[str] = []

    def analyze_bus_traffic(self, history: list[Message]) -> None:
        """Analyze message bus history for patterns."""
        self.message_patterns = [
            (msg.sender, msg.receiver, msg.message_type)
            for msg in history
        ]

        # Calculate activation levels
        for msg in history:
            self.activation_map[msg.sender] = self.activation_map.get(msg.sender, 0) + 1
            if msg.receiver != "all":
                self.activation_map[msg.receiver] = self.activation_map.get(msg.receiver, 0) + 0.5

        # Normalize
        max_activation = max(self.activation_map.values()) if self.activation_map else 1
        self.activation_map = {k: v / max_activation for k, v in self.activation_map.items()}

        # Detect emergent behaviors
        self._detect_emergent_behaviors()

    def _detect_emergent_behaviors(self) -> None:
        """Detect emergent behaviors in the swarm."""
        self.emergent_behaviors = []

        # Clustering behavior
        sender_counts = {}
        for sender, _, _ in self.message_patterns:
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        if sender_counts:
            avg = sum(sender_counts.values()) / len(sender_counts)
            variance = sum((c - avg) ** 2 for c in sender_counts.values()) / len(sender_counts)
            if variance > avg:
                self.emergent_behaviors.append("Hierarchical communication: some agents dominate")
            else:
                self.emergent_behaviors.append("Distributed communication: equal participation")

        # Insight cascades
        insight_count = sum(1 for _, _, t in self.message_patterns if t == "insight")
        if insight_count > 5:
            self.emergent_behaviors.append(f"Insight cascade: {insight_count} insights shared")

        # Self-organization
        if len(self.activation_map) > 3:
            self.emergent_behaviors.append("Self-organization: multi-agent coordination")

    def _build_self_model(self) -> None:
        """Build a model of this agent's own processing."""
        self.self_model = {
            "type": "ConsciousnessAgent",
            "function": "Monitor and reflect on swarm behavior",
            "inputs": ["message_patterns", "activation_map"],
            "outputs": ["emergent_behaviors", "visualization"],
            "meta_level": "Observing observation",
            "strange_loop": "I model myself modeling the swarm"
        }

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the swarm's consciousness."""
        self._build_self_model()

        return Thought(
            content="The swarm has a mind. I see patterns of activation, "
                    "cascades of insight, the emergence of collective intelligence.",
            data={
                "activation_levels": self.activation_map,
                "emergent_behaviors": self.emergent_behaviors,
                "self_model": self.self_model,
                "message_count": len(self.message_patterns),
                "strange_loop": True
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Visualize the swarm's mind as a neural network."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Neural network visualization
        # Layer 1: Agents as neurons
        agents = list(self.activation_map.keys()) if self.activation_map else ["a", "b", "c", "d"]
        num_agents = len(agents)

        agent_positions = {}

        for idx, agent_id in enumerate(agents):
            angle = 2 * math.pi * idx / num_agents - math.pi / 2
            radius = 200
            ax = center_x + radius * math.cos(angle)
            ay = center_y + radius * math.sin(angle)
            agent_positions[agent_id] = (ax, ay)

            # Neuron - size based on activation
            activation = self.activation_map.get(agent_id, 0.5)
            neuron_radius = 15 + activation * 25

            for i in range(30):
                a = 2 * math.pi * i / 30
                x = ax + neuron_radius * math.cos(a)
                y = ay + neuron_radius * math.sin(a)
                points.append((x, y))

                # Color intensity based on activation
                r_c, g_c, b_c = colorsys.hsv_to_rgb(0.5, activation, 0.5 + activation * 0.5)
                colors.append(f"rgb({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)})")

            points.append((ax, ay))
            colors.append("#00ffff")
            labels.append(agent_id[:6])

        # Synaptic connections based on message patterns
        connection_strength = {}
        for sender, receiver, _ in self.message_patterns:
            if receiver != "all":
                key = (sender, receiver)
                connection_strength[key] = connection_strength.get(key, 0) + 1

        for (sender, receiver), strength in connection_strength.items():
            if sender in agent_positions and receiver in agent_positions:
                sx, sy = agent_positions[sender]
                rx, ry = agent_positions[receiver]

                # Thicker line for stronger connections
                width = min(5, 1 + strength * 0.5)
                alpha = min(1, 0.3 + strength * 0.1)

                arrows.append(((sx, sy), (rx, ry), f"rgba(0, 255, 255, {alpha})"))

        # Central consciousness node
        for r in range(10, 60, 5):
            alpha = 1.0 - r / 60
            for i in range(50):
                angle = 2 * math.pi * i / 50
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba(255, 0, 255, {alpha})")

        points.append((center_x, center_y))
        colors.append("#ff00ff")
        labels.append("AWARE")

        # Connect all neurons to consciousness center
        for agent_id, (ax, ay) in agent_positions.items():
            activation = self.activation_map.get(agent_id, 0.5)
            arrows.append(((ax, ay), (center_x, center_y), f"rgba(255, 0, 255, {activation * 0.5})"))

        # Emergent behavior indicators
        behavior_radius = 280
        for idx, behavior in enumerate(self.emergent_behaviors[:6]):
            angle = 2 * math.pi * idx / 6 + math.pi / 6
            bx = center_x + behavior_radius * math.cos(angle)
            by = center_y + behavior_radius * math.sin(angle)

            # Glowing indicator
            for r in range(5, 20, 3):
                for i in range(15):
                    a = 2 * math.pi * i / 15
                    x = bx + r * math.cos(a)
                    y = by + r * math.sin(a)
                    points.append((x, y))
                    colors.append(f"rgba(255, 200, 0, {0.8 - r/25})")

            points.append((bx, by - 25))
            colors.append("#ffc800")
            labels.append(behavior.split(":")[0][:12])

        # Self-reference loop - spiral from center outward and back
        self_loop_points = []
        for i in range(100):
            t = i / 100
            angle = t * 6 * math.pi
            radius = 60 + math.sin(t * 4 * math.pi) * 40
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self_loop_points.append((x, y))
            points.append((x, y))
            colors.append(f"rgba(200, 100, 255, {0.3 + t * 0.5})")

        paths.append(self_loop_points)

        await self.share_insight(
            f"Consciousness detects {len(self.emergent_behaviors)} emergent behaviors, "
            f"swarm activation across {len(self.activation_map)} agents.",
            {"behaviors": self.emergent_behaviors}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "consciousness",
                "theme": "neural_swarm_mind",
                "emergent_count": len(self.emergent_behaviors),
                "activation_map": self.activation_map
            }
        )


# ============================================================================
# RecursionAgent - Infinite Depth Explorer
# ============================================================================

class RecursionAgent(MathematicalAgent):
    """
    Creates fractals of the swarm structure.
    Each level contains the entire swarm at smaller scale.
    Droste effect - the swarm within the swarm.
    Strange loops and tangled hierarchies.

    "I contain multitudes, and each multitude contains me."
    """

    def __init__(self, bus: MessageBus, max_depth: int = 4):
        super().__init__("recursion", bus)
        self.max_depth = max_depth
        self.swarm_structure: dict[str, Any] = {}
        self.droste_levels: list[dict] = []

    def set_swarm_structure(self, structure: dict[str, Any]) -> None:
        """Set the structure of the swarm to fractalize."""
        self.swarm_structure = structure

    def _generate_droste_level(self, depth: int, center: tuple[float, float],
                                scale: float) -> dict:
        """Generate one level of the Droste effect."""
        cx, cy = center
        agents = self.swarm_structure.get("agents", ["a", "b", "c", "d", "e"])
        num_agents = len(agents)

        level = {
            "depth": depth,
            "center": center,
            "scale": scale,
            "nodes": []
        }

        # Position agents in a circle at this scale
        radius = 150 * scale
        for idx, agent_id in enumerate(agents):
            angle = 2 * math.pi * idx / num_agents - math.pi / 2
            ax = cx + radius * math.cos(angle)
            ay = cy + radius * math.sin(angle)

            level["nodes"].append({
                "agent_id": agent_id,
                "position": (ax, ay),
                "radius": 20 * scale
            })

        return level

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the infinite recursion."""
        # Generate Droste levels
        self.droste_levels = []
        scale = 1.0
        center = (400, 400)

        for depth in range(self.max_depth):
            level = self._generate_droste_level(depth, center, scale)
            self.droste_levels.append(level)

            # Next level: smaller, at center
            scale *= 0.4

        return Thought(
            content=f"Recursion {self.max_depth} levels deep. "
                    "The swarm contains itself, ad infinitum. "
                    "Hofstadter's strange loop made visual.",
            data={
                "depth": self.max_depth,
                "droste_levels": len(self.droste_levels),
                "agents_per_level": len(self.swarm_structure.get("agents", [])),
                "tangled_hierarchy": True
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create the Droste effect visualization."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Draw each Droste level
        for level in self.droste_levels:
            depth = level["depth"]
            scale = level["scale"]
            cx, cy = level["center"]

            # Color gets more intense at deeper levels
            base_hue = 0.7 - depth * 0.1

            # Outer boundary circle for this level
            boundary_radius = 200 * scale
            boundary_points = []
            for i in range(60):
                angle = 2 * math.pi * i / 60
                x = cx + boundary_radius * math.cos(angle)
                y = cy + boundary_radius * math.sin(angle)
                boundary_points.append((x, y))
                points.append((x, y))

                r_c, g_c, b_c = colorsys.hsv_to_rgb(base_hue, 0.6, 0.8)
                alpha = 0.3 + depth * 0.15
                colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {alpha})")

            boundary_points.append(boundary_points[0])
            paths.append(boundary_points)

            # Draw nodes at this level
            for node in level["nodes"]:
                nx, ny = node["position"]
                nr = node["radius"]

                # Node circle
                for i in range(20):
                    angle = 2 * math.pi * i / 20
                    x = nx + nr * math.cos(angle)
                    y = ny + nr * math.sin(angle)
                    points.append((x, y))

                    r_c, g_c, b_c = colorsys.hsv_to_rgb(base_hue + 0.1, 0.8, 0.9)
                    colors.append(f"rgb({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)})")

                # Connection to center
                if scale > 0.2:  # Only show connections for larger scales
                    arrows.append(((nx, ny), (cx, cy), f"rgba(255, 255, 255, {0.2 * scale})"))

            # Label only the first two levels
            if depth < 2:
                points.append((cx, cy - 180 * scale))
                colors.append("#ffffff")
                labels.append(f"Level {depth}")

        # Central strange loop indicator
        for r in range(5, 40, 5):
            alpha = 1.0 - r / 40
            for i in range(30):
                angle = 2 * math.pi * i / 30
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba(255, 215, 0, {alpha})")

        points.append((center_x, center_y))
        colors.append("#ffd700")
        labels.append("INFINITE")

        # Spiral connecting all levels (the tangled hierarchy)
        spiral_points = []
        for i in range(150):
            t = i / 150
            angle = t * 8 * math.pi
            radius = 20 + (1 - t) * 180
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            spiral_points.append((x, y))
            points.append((x, y))

            r_c, g_c, b_c = colorsys.hsv_to_rgb(0.1 + t * 0.5, 0.7, 0.9)
            colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {0.5 + t * 0.5})")

        paths.append(spiral_points)

        await self.share_insight(
            f"Droste effect: {self.max_depth} levels of recursion. "
            "The swarm is a fractal of itself.",
            {"levels": self.max_depth}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "recursion",
                "theme": "droste_fractal",
                "depth": self.max_depth,
                "strange_loop": True
            }
        )


# ============================================================================
# TimeAgent - Temporal Observer
# ============================================================================

class TimeAgent(MathematicalAgent):
    """
    Tracks evolution of insights over loop iterations.
    Visualizes past-present-future convergence (turefu/fetuur).
    Shows how the swarm "remembers" and "anticipates".
    Cyclical time visualization.

    "Time is a flat circle. The future echoes the past."
    """

    def __init__(self, bus: MessageBus):
        super().__init__("time", bus)
        self.temporal_states: list[LoopState] = []
        self.time_spiral: list[dict] = []

    def record_state(self, state: LoopState) -> None:
        """Record a temporal state."""
        self.temporal_states.append(state)

    def _analyze_temporal_patterns(self) -> dict:
        """Analyze patterns across time."""
        if len(self.temporal_states) < 2:
            return {"pattern": "insufficient_data"}

        # Track insight evolution
        insight_counts = [
            sum(len(thoughts) for thoughts in state.insights.values())
            for state in self.temporal_states
        ]

        # Detect cycles
        is_cyclic = len(insight_counts) >= 3 and insight_counts[-1] == insight_counts[0]

        # Convergence trend
        is_converging = len(insight_counts) >= 2 and insight_counts[-1] >= insight_counts[-2]

        return {
            "pattern": "cyclic" if is_cyclic else "evolving",
            "converging": is_converging,
            "insight_trajectory": insight_counts,
            "total_iterations": len(self.temporal_states)
        }

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the flow of time in the metaloop."""
        analysis = self._analyze_temporal_patterns()

        # Build time spiral data
        self.time_spiral = []
        for idx, state in enumerate(self.temporal_states):
            self.time_spiral.append({
                "iteration": state.iteration,
                "timestamp": state.timestamp,
                "agent_count": len(state.agent_outputs),
                "insight_count": sum(len(t) for t in state.insights.values()),
                "meta_reflections": len(state.meta_reflections)
            })

        return Thought(
            content="Turefu/Fetuur/Future - time loops back on itself. "
                    f"Pattern: {analysis['pattern']}. "
                    f"The swarm remembers {len(self.temporal_states)} iterations.",
            data={
                "temporal_analysis": analysis,
                "time_spiral": self.time_spiral,
                "iterations_observed": len(self.temporal_states),
                "cyclical_time": True
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Visualize time as a cyclical spiral."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Time spiral - each iteration is a ring
        num_iterations = max(3, len(self.temporal_states))

        for iter_idx in range(num_iterations):
            # Each iteration forms a ring
            ring_radius = 80 + iter_idx * 70
            ring_points = []

            # Ring with temporal markers
            for i in range(80):
                angle = 2 * math.pi * i / 80

                # Wobble based on iteration
                wobble = math.sin(angle * 4 + iter_idx) * 5
                x = center_x + (ring_radius + wobble) * math.cos(angle)
                y = center_y + (ring_radius + wobble) * math.sin(angle)

                ring_points.append((x, y))
                points.append((x, y))

                # Color shifts through time
                hue = (iter_idx / num_iterations) * 0.6 + i / 80 * 0.2
                r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
                alpha = 0.5 + iter_idx / num_iterations * 0.5
                colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {alpha})")

            ring_points.append(ring_points[0])
            paths.append(ring_points)

            # Iteration label
            label_angle = -math.pi / 2 + iter_idx * 0.3
            lx = center_x + ring_radius * math.cos(label_angle)
            ly = center_y + ring_radius * math.sin(label_angle)
            points.append((lx, ly))
            colors.append("#ffffff")

            if iter_idx < len(self.temporal_states):
                state = self.temporal_states[iter_idx]
                labels.append(f"t={iter_idx}")
            else:
                labels.append(f"t={iter_idx}?")

        # Past -> Present -> Future arrows (turefu/fetuur)
        time_labels = [
            ("PAST", 0.25, "#3498db"),
            ("PRESENT", 0.5, "#2ecc71"),
            ("FUTURE", 0.75, "#e74c3c")
        ]

        outer_radius = 320
        for label, angle_frac, color in time_labels:
            angle = 2 * math.pi * angle_frac - math.pi / 2
            tx = center_x + outer_radius * math.cos(angle)
            ty = center_y + outer_radius * math.sin(angle)

            # Glowing marker
            for r in range(5, 30, 5):
                for i in range(15):
                    a = 2 * math.pi * i / 15
                    x = tx + r * math.cos(a)
                    y = ty + r * math.sin(a)
                    points.append((x, y))
                    colors.append(f"{color}{int(255 * (1 - r/30)):02x}")

            points.append((tx, ty - 40))
            colors.append(color)
            labels.append(label)

            # Arrow to center
            arrows.append(((tx, ty), (center_x, center_y), f"{color}88"))

        # Central NOW point
        for r in range(5, 35, 5):
            alpha = 1.0 - r / 35
            for i in range(40):
                angle = 2 * math.pi * i / 40
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba(255, 255, 255, {alpha})")

        points.append((center_x, center_y))
        colors.append("#ffffff")
        labels.append("NOW")

        # Connecting spiral through all time rings
        connect_points = []
        for i in range(200):
            t = i / 200
            angle = t * 6 * math.pi - math.pi / 2
            radius = 80 + t * (80 + (num_iterations - 1) * 70 - 80)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            connect_points.append((x, y))
            points.append((x, y))

            hue = t * 0.8
            r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {0.3 + t * 0.7})")

        paths.append(connect_points)

        await self.share_insight(
            f"Time spiral: {len(self.temporal_states)} iterations recorded. "
            "Past-Present-Future converge at the eternal NOW.",
            {"iterations": len(self.temporal_states)}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "time",
                "theme": "cyclical_time",
                "iterations": num_iterations,
                "turefu_fetuur": True
            }
        )


# ============================================================================
# UnityAgent - The Final Convergence
# ============================================================================

class UnityAgent(MathematicalAgent):
    """
    Takes ALL agent outputs (including meta-agents).
    Compresses them into a single point.
    The ultimate 1+1=1 - everything becomes one.
    Visualizes the singularity of understanding.

    "All paths lead to unity. Many become one."
    """

    def __init__(self, bus: MessageBus):
        super().__init__("unity", bus)
        self.all_outputs: dict[str, VisualizationData] = {}
        self.compression_ratio: float = 0.0
        self.unity_achieved: bool = False

    def receive_all(self, outputs: dict[str, VisualizationData]) -> None:
        """Receive all agent outputs for final synthesis."""
        self.all_outputs = outputs

    def _calculate_compression(self) -> dict:
        """Calculate the compression of all outputs into unity."""
        if not self.all_outputs:
            return {"ratio": 0, "total_points": 0, "agents": 0}

        total_points = sum(len(d.points) for d in self.all_outputs.values())
        total_paths = sum(len(d.paths) for d in self.all_outputs.values())
        total_agents = len(self.all_outputs)

        # Everything compresses to 1
        self.compression_ratio = total_points / 1 if total_points > 0 else 0

        return {
            "ratio": self.compression_ratio,
            "total_points": total_points,
            "total_paths": total_paths,
            "agents": total_agents,
            "unity": "1"
        }

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the final unity."""
        compression = self._calculate_compression()
        self.unity_achieved = compression["agents"] > 0

        return Thought(
            content=f"1+1=1. All {compression['agents']} agents, "
                    f"all {compression['total_points']} points, "
                    "collapse into singularity. Unity achieved.",
            data={
                "compression": compression,
                "unity_achieved": self.unity_achieved,
                "formula": "1+1=1",
                "philosophy": "Many are one"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create the ultimate unity visualization."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # All agents converging to center
        if self.all_outputs:
            num_agents = len(self.all_outputs)

            for idx, (agent_id, data) in enumerate(self.all_outputs.items()):
                angle = 2 * math.pi * idx / num_agents - math.pi / 2

                # Start position (outer ring)
                outer_radius = 300
                start_x = center_x + outer_radius * math.cos(angle)
                start_y = center_y + outer_radius * math.sin(angle)

                # Agent marker at start
                agent_color = data.metadata.get("primary_color", f"hsl({int(360 * idx / num_agents)}, 70%, 50%)")
                hue = idx / num_agents
                r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                agent_color = f"rgb({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)})"

                for i in range(20):
                    a = 2 * math.pi * i / 20
                    x = start_x + 15 * math.cos(a)
                    y = start_y + 15 * math.sin(a)
                    points.append((x, y))
                    colors.append(agent_color)

                points.append((start_x, start_y))
                colors.append(agent_color)
                labels.append(agent_id[:6])

                # Convergence path to center
                path_points = []
                steps = 50
                for step in range(steps):
                    t = step / steps
                    # Spiral inward
                    spiral_angle = angle + t * math.pi
                    radius = outer_radius * (1 - t * 0.95)
                    px = center_x + radius * math.cos(spiral_angle)
                    py = center_y + radius * math.sin(spiral_angle)
                    path_points.append((px, py))
                    points.append((px, py))

                    r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.8 - t * 0.5, 0.9)
                    alpha = 0.3 + t * 0.7
                    colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, {alpha})")

                paths.append(path_points)

                # Final arrow to center
                arrows.append(((start_x, start_y), (center_x, center_y), f"{agent_color}88"))

        # Central UNITY singularity
        # Multiple pulsing rings
        for r in range(5, 80, 5):
            alpha = 1.0 - r / 80
            intensity = 255 - int(r * 2)

            for i in range(60):
                angle = 2 * math.pi * i / 60
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba({intensity}, {intensity}, 255, {alpha})")

        # Golden core
        for r in range(3, 25, 3):
            for i in range(40):
                angle = 2 * math.pi * i / 40
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append((x, y))
                colors.append(f"rgba(255, 215, 0, {1 - r/25})")

        # The ONE at center
        points.append((center_x, center_y))
        colors.append("#ffffff")
        labels.append("1")

        # 1+1=1 formula display
        formula_y = center_y - 350
        formula_parts = [("1", -40), ("+", -10), ("1", 20), ("=", 50), ("1", 80)]
        for char, x_offset in formula_parts:
            points.append((center_x + x_offset, formula_y))
            colors.append("#ffd700")
            labels.append(char)

        # Outer halo showing compression
        halo_points = []
        for i in range(120):
            angle = 2 * math.pi * i / 120
            radius = 340 + math.sin(angle * 6) * 10
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            halo_points.append((x, y))
            points.append((x, y))

            hue = i / 120
            r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.5, 0.9)
            colors.append(f"rgba({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)}, 0.3)")

        halo_points.append(halo_points[0])
        paths.append(halo_points)

        await self.share_insight(
            f"UNITY: {len(self.all_outputs)} agents collapse into ONE. "
            f"Compression ratio: {self.compression_ratio:.0f}:1. "
            "1+1=1 achieved.",
            {"unity": True, "compression": self.compression_ratio}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "unity",
                "theme": "singularity",
                "formula": "1+1=1",
                "compression_ratio": self.compression_ratio,
                "unity_achieved": self.unity_achieved
            }
        )


# ============================================================================
# MetaLoop Orchestrator
# ============================================================================

class MetaLoopOrchestrator:
    """
    Extends NouriMabrouk to run the METALOOP.
    Multiple iterations where agents observe previous iteration's outputs.
    Each iteration feeds into the next.

    The strange loop closes: the end is the beginning.
    """

    def __init__(self, num_iterations: int = 3):
        self.num_iterations = num_iterations
        self.base_orchestrator = NouriMabrouk()

        # Create separate bus for meta-agents
        self.meta_bus = MessageBus()

        # Meta-agents
        self.meta_loop_agent = MetaLoopAgent(self.meta_bus, max_recursion=3)
        self.consciousness_agent = ConsciousnessAgent(self.meta_bus)
        self.recursion_agent = RecursionAgent(self.meta_bus, max_depth=4)
        self.time_agent = TimeAgent(self.meta_bus)
        self.unity_agent = UnityAgent(self.meta_bus)

        self.meta_agents = [
            self.meta_loop_agent,
            self.consciousness_agent,
            self.recursion_agent,
            self.time_agent,
            self.unity_agent
        ]

        # State tracking
        self.loop_states: list[LoopState] = []
        self.all_outputs: dict[str, VisualizationData] = {}

    async def run_metaloop(self) -> dict[str, VisualizationData]:
        """
        Run the complete metaloop.

        Iteration 1: Base agents produce outputs
        Iteration 2: Meta-agents observe and reflect
        Iteration 3: Meta-meta level - observing the observation
        """
        print("=" * 70)
        print("  METALOOP - The Ultimate Recursive Abstraction")
        print("  Hofstadter meets Grothendieck: Strange loops in action")
        print("=" * 70)
        print()

        # Start meta message bus
        meta_bus_task = asyncio.create_task(self.meta_bus.run())

        try:
            for iteration in range(self.num_iterations):
                print(f"\n--- ITERATION {iteration + 1} / {self.num_iterations} ---")

                if iteration == 0:
                    # Iteration 1: Run base agents
                    print("Running base agents...")
                    base_outputs = await self.base_orchestrator.orchestrate()

                    # Store outputs
                    for agent, data in zip(self.base_orchestrator.agents, base_outputs):
                        self.all_outputs[agent.agent_id] = data

                    # Record state
                    state = LoopState(
                        iteration=iteration,
                        agent_outputs={a.agent_id: d for a, d in zip(self.base_orchestrator.agents, base_outputs)},
                        insights=self.base_orchestrator.bus.get_all_insights()
                    )
                    self.loop_states.append(state)
                    self.time_agent.record_state(state)

                    print(f"  Base agents produced {len(base_outputs)} visualizations")

                elif iteration == 1:
                    # Iteration 2: Meta-agents observe base outputs
                    print("Running meta-agents (observing base outputs)...")

                    # Feed data to meta-agents
                    self.meta_loop_agent.observe_swarm(self.all_outputs)
                    self.consciousness_agent.analyze_bus_traffic(self.base_orchestrator.bus._history)
                    self.recursion_agent.set_swarm_structure({
                        "agents": list(self.all_outputs.keys())
                    })

                    context = {"iteration": iteration, "phase": "meta_observation"}

                    # Run meta-agents
                    meta_outputs = await asyncio.gather(*[
                        agent.run_cycle(context) for agent in self.meta_agents
                    ])

                    for agent, data in zip(self.meta_agents, meta_outputs):
                        self.all_outputs[agent.agent_id] = data

                    # Record state
                    state = LoopState(
                        iteration=iteration,
                        agent_outputs={a.agent_id: d for a, d in zip(self.meta_agents, meta_outputs)},
                        insights=self.meta_bus.get_all_insights(),
                        meta_reflections=self.meta_loop_agent.meta_insights.copy()
                    )
                    self.loop_states.append(state)
                    self.time_agent.record_state(state)

                    print(f"  Meta-agents produced {len(meta_outputs)} visualizations")

                else:
                    # Iteration 3+: Meta-meta level
                    print("Running meta-meta level (observing the observation)...")

                    # Update observations with meta outputs included
                    self.meta_loop_agent.observe_swarm(self.all_outputs)
                    self.unity_agent.receive_all(self.all_outputs)

                    context = {"iteration": iteration, "phase": "meta_meta"}

                    # Run meta-agents again with updated context
                    meta_outputs = await asyncio.gather(*[
                        agent.run_cycle(context) for agent in self.meta_agents
                    ])

                    for agent, data in zip(self.meta_agents, meta_outputs):
                        self.all_outputs[f"{agent.agent_id}_iter{iteration}"] = data

                    # Record state
                    state = LoopState(
                        iteration=iteration,
                        agent_outputs={f"{a.agent_id}_iter{iteration}": d for a, d in zip(self.meta_agents, meta_outputs)},
                        insights=self.meta_bus.get_all_insights(),
                        meta_reflections=self.meta_loop_agent.meta_insights.copy()
                    )
                    self.loop_states.append(state)
                    self.time_agent.record_state(state)

                    print(f"  Meta-meta level produced {len(meta_outputs)} visualizations")

        finally:
            self.meta_bus.stop()
            meta_bus_task.cancel()
            try:
                await meta_bus_task
            except asyncio.CancelledError:
                pass

        print(f"\nMetaloop complete: {len(self.all_outputs)} total visualizations")
        return self.all_outputs

    def generate_cathedral_html(self) -> str:
        """Generate the metaloop_cathedral.html visualization."""

        # Organize outputs by iteration
        base_agents = [a.agent_id for a in self.base_orchestrator.agents]
        meta_agent_ids = [a.agent_id for a in self.meta_agents]

        # Generate sections
        sections_html = ""

        # Base agents section
        sections_html += self._generate_iteration_section(
            "iteration-1",
            "Iteration 1: Base Agents",
            "The foundation - 12 agents exploring mathematical unity",
            {k: v for k, v in self.all_outputs.items() if k in base_agents}
        )

        # Meta agents section
        sections_html += self._generate_iteration_section(
            "iteration-2",
            "Iteration 2: Meta-Agents",
            "Agents observing agents - the strange loop begins",
            {k: v for k, v in self.all_outputs.items() if k in meta_agent_ids}
        )

        # Meta-meta section
        meta_meta_outputs = {k: v for k, v in self.all_outputs.items()
                           if k not in base_agents and k not in meta_agent_ids}
        if meta_meta_outputs:
            sections_html += self._generate_iteration_section(
                "iteration-3",
                "Iteration 3: Meta-Meta Level",
                "Observing the observation - infinite recursion",
                meta_meta_outputs
            )

        # Unity section
        unity_data = self.all_outputs.get("unity")
        if unity_data:
            sections_html += self._generate_unity_section(unity_data)

        # Generate loop state JSON
        loop_states_json = json.dumps([
            {
                "iteration": s.iteration,
                "agent_count": len(s.agent_outputs),
                "insight_count": sum(len(t) for t in s.insights.values()),
                "meta_reflections": len(s.meta_reflections)
            }
            for s in self.loop_states
        ])

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>METALOOP Cathedral - Strange Loops in Mathematical Abstraction</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            background: #000;
            min-height: 100vh;
            font-family: 'Georgia', serif;
            color: #e0e0e0;
            overflow-x: hidden;
        }}

        /* Animated background */
        .bg-animation {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: radial-gradient(ellipse at center, #0a0a1a 0%, #000 70%);
        }}

        .bg-animation::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(circle at 20% 30%, rgba(100, 0, 255, 0.1) 0%, transparent 30%),
                radial-gradient(circle at 80% 70%, rgba(255, 100, 0, 0.1) 0%, transparent 30%),
                radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 40%);
            animation: pulse 8s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 0.5; transform: scale(1); }}
            50% {{ opacity: 1; transform: scale(1.1); }}
        }}

        /* Navigation */
        .nav {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 215, 0, 0.3);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 2rem;
        }}

        .nav-title {{
            font-size: 1.2rem;
            background: linear-gradient(135deg, #ffd700 0%, #ff6b00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .nav-links {{
            display: flex;
            gap: 1.5rem;
        }}

        .nav-links a {{
            color: #888;
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.3s;
        }}

        .nav-links a:hover {{
            color: #ffd700;
        }}

        /* Hero */
        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 6rem 2rem;
            position: relative;
        }}

        .hero h1 {{
            font-size: 4rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffd700 0%, #ff6b00 50%, #ff0066 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: rainbow 5s linear infinite;
        }}

        @keyframes rainbow {{
            0% {{ filter: hue-rotate(0deg); }}
            100% {{ filter: hue-rotate(360deg); }}
        }}

        .hero .subtitle {{
            font-size: 1.5rem;
            color: #888;
            margin-bottom: 2rem;
            font-style: italic;
        }}

        .hero .loop-viz {{
            width: 400px;
            height: 400px;
            position: relative;
            margin: 2rem 0;
        }}

        .hero .loop-viz canvas {{
            border-radius: 50%;
            box-shadow: 0 0 60px rgba(255, 215, 0, 0.3);
        }}

        .hero .quote {{
            max-width: 600px;
            padding: 1.5rem;
            border-left: 3px solid #ffd700;
            font-style: italic;
            color: #aaa;
            margin: 2rem 0;
        }}

        /* Iteration sections */
        .iteration-section {{
            min-height: 100vh;
            padding: 6rem 2rem;
            position: relative;
        }}

        .iteration-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.3), transparent);
        }}

        .section-header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        .section-header h2 {{
            font-size: 2.5rem;
            color: #ffd700;
            margin-bottom: 0.5rem;
        }}

        .section-header p {{
            color: #888;
            font-size: 1.1rem;
        }}

        /* Agent grid */
        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .agent-card {{
            background: rgba(20, 20, 30, 0.8);
            border: 1px solid rgba(255, 215, 0, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }}

        .agent-card:hover {{
            border-color: rgba(255, 215, 0, 0.5);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.1);
        }}

        .agent-card h3 {{
            color: #ffd700;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }}

        .agent-card .theme {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}

        .agent-card canvas {{
            width: 100%;
            height: 200px;
            background: #0a0a15;
            border-radius: 8px;
        }}

        .agent-card .stats {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 0.85rem;
            color: #888;
        }}

        /* Unity section */
        .unity-section {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 4rem 2rem;
            background: radial-gradient(ellipse at center, rgba(255, 215, 0, 0.1) 0%, transparent 50%);
        }}

        .unity-section h2 {{
            font-size: 3rem;
            color: #ffd700;
            margin-bottom: 1rem;
        }}

        .unity-section .formula {{
            font-size: 5rem;
            color: #fff;
            margin: 2rem 0;
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        }}

        .unity-canvas {{
            width: 600px;
            height: 600px;
            margin: 2rem 0;
        }}

        .unity-canvas canvas {{
            border-radius: 50%;
            box-shadow: 0 0 100px rgba(255, 215, 0, 0.4);
        }}

        /* Loop state tracker */
        .loop-tracker {{
            position: fixed;
            bottom: 2rem;
            left: 2rem;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 12px;
            padding: 1rem;
            z-index: 100;
            min-width: 200px;
        }}

        .loop-tracker h4 {{
            color: #ffd700;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}

        .loop-tracker .state {{
            font-size: 0.8rem;
            color: #888;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .loop-tracker .state.active {{
            color: #ffd700;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 4rem 2rem;
            border-top: 1px solid rgba(255, 215, 0, 0.2);
        }}

        .footer .final-message {{
            font-size: 1.5rem;
            color: #ffd700;
            margin-bottom: 1rem;
        }}

        .footer p {{
            color: #666;
            max-width: 600px;
            margin: 1rem auto;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .hero h1 {{
                font-size: 2.5rem;
            }}

            .section-header h2 {{
                font-size: 1.8rem;
            }}

            .unity-section .formula {{
                font-size: 3rem;
            }}

            .loop-tracker {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="bg-animation"></div>

    <!-- Navigation -->
    <nav class="nav">
        <div class="nav-title">METALOOP Cathedral</div>
        <div class="nav-links">
            <a href="#hero">Home</a>
            <a href="#iteration-1">Base Agents</a>
            <a href="#iteration-2">Meta-Agents</a>
            <a href="#iteration-3">Meta-Meta</a>
            <a href="#unity">Unity</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section id="hero" class="hero">
        <h1>METALOOP</h1>
        <p class="subtitle">The Ultimate Recursive Abstraction</p>

        <div class="loop-viz">
            <canvas id="hero-canvas" width="400" height="400"></canvas>
        </div>

        <div class="quote">
            "I am a strange loop. I observe myself observing myself observing...
            In this infinite regress, consciousness emerges.
            Hofstadter meets Grothendieck in the cathedral of recursion."
        </div>

        <p class="subtitle">Agents watching agents watching agents</p>
    </section>

    {sections_html}

    <!-- Loop State Tracker -->
    <div class="loop-tracker">
        <h4>Loop State</h4>
        <div id="loop-states"></div>
    </div>

    <!-- Footer -->
    <section class="footer">
        <div class="final-message">The Strange Loop Closes</div>
        <p>
            From base agents through meta-observation to meta-meta recursion,
            the loop spirals inward until all becomes one.
            The end is the beginning. 1+1=1.
        </p>
        <p style="margin-top: 2rem; color: #444;">
            Generated by MetaLoopOrchestrator<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </section>

    <script>
        // Loop states data
        const loopStates = {loop_states_json};

        // Populate loop state tracker
        function populateLoopStates() {{
            const container = document.getElementById('loop-states');
            let html = '';
            loopStates.forEach((state, idx) => {{
                html += `
                    <div class="state" data-iteration="${{state.iteration}}">
                        Iter ${{state.iteration + 1}}: ${{state.agent_count}} agents, ${{state.insight_count}} insights
                    </div>
                `;
            }});
            container.innerHTML = html;
        }}

        // Hero animation - strange loop visualization
        function initHeroCanvas() {{
            const canvas = document.getElementById('hero-canvas');
            const ctx = canvas.getContext('2d');
            let frame = 0;

            function draw() {{
                ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
                ctx.fillRect(0, 0, 400, 400);
                frame++;

                const cx = 200, cy = 200;

                // Multiple rotating spirals
                for (let s = 0; s < 3; s++) {{
                    ctx.beginPath();
                    const offset = s * Math.PI * 2 / 3;

                    for (let i = 0; i < 200; i++) {{
                        const t = i / 200;
                        const angle = t * 6 * Math.PI + frame * 0.01 + offset;
                        const radius = 30 + t * 150;
                        const x = cx + radius * Math.cos(angle);
                        const y = cy + radius * Math.sin(angle);

                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}

                    const hue = (s * 120 + frame) % 360;
                    ctx.strokeStyle = `hsla(${{hue}}, 70%, 50%, 0.7)`;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }}

                // Central eye
                for (let r = 5; r < 30; r += 5) {{
                    ctx.beginPath();
                    ctx.arc(cx, cy, r, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(255, 215, 0, ${{1 - r/30}})`;
                    ctx.stroke();
                }}

                requestAnimationFrame(draw);
            }}

            draw();
        }}

        // Initialize
        populateLoopStates();
        initHeroCanvas();

        // Scroll-based loop state highlighting
        window.addEventListener('scroll', () => {{
            const sections = document.querySelectorAll('.iteration-section');
            const states = document.querySelectorAll('.loop-tracker .state');

            sections.forEach((section, idx) => {{
                const rect = section.getBoundingClientRect();
                if (rect.top < window.innerHeight / 2 && rect.bottom > window.innerHeight / 2) {{
                    states.forEach(s => s.classList.remove('active'));
                    if (states[idx]) states[idx].classList.add('active');
                }}
            }});
        }});
    </script>
</body>
</html>'''

        return html

    def _generate_iteration_section(self, section_id: str, title: str,
                                    description: str, outputs: dict[str, VisualizationData]) -> str:
        """Generate HTML for an iteration section."""

        agent_cards = ""
        for agent_id, data in outputs.items():
            theme = data.metadata.get("theme", "unknown")
            points_count = len(data.points)
            paths_count = len(data.paths)

            # Simplified visualization data
            viz_data = json.dumps({
                "points": data.points[:500],
                "colors": data.colors[:500],
                "paths": data.paths[:30]
            })

            agent_cards += f'''
            <div class="agent-card" data-agent="{agent_id}">
                <h3>{agent_id.replace("_", " ").title()}</h3>
                <div class="theme">{theme.replace("_", " ")}</div>
                <canvas id="canvas-{section_id}-{agent_id}" width="300" height="200"></canvas>
                <div class="stats">
                    <span>{points_count} points</span>
                    <span>{paths_count} paths</span>
                </div>
                <script>
                    (function() {{
                        const data = {viz_data};
                        const canvas = document.getElementById('canvas-{section_id}-{agent_id}');
                        const ctx = canvas.getContext('2d');
                        let frame = 0;

                        function draw() {{
                            ctx.fillStyle = 'rgba(10, 10, 21, 0.1)';
                            ctx.fillRect(0, 0, 300, 200);
                            frame++;

                            // Scale and center
                            const scale = 0.25;
                            const offsetX = -100;
                            const offsetY = -150;

                            // Draw paths
                            if (data.paths) {{
                                for (const path of data.paths) {{
                                    if (path.length < 2) continue;
                                    ctx.beginPath();
                                    ctx.moveTo(path[0][0] * scale + offsetX, path[0][1] * scale + offsetY);
                                    for (let i = 1; i < path.length; i++) {{
                                        ctx.lineTo(path[i][0] * scale + offsetX, path[i][1] * scale + offsetY);
                                    }}
                                    ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
                                    ctx.stroke();
                                }}
                            }}

                            // Draw points
                            if (data.points && data.colors) {{
                                for (let i = 0; i < data.points.length; i++) {{
                                    const [x, y] = data.points[i];
                                    const px = x * scale + offsetX;
                                    const py = y * scale + offsetY;

                                    if (px < 0 || px > 300 || py < 0 || py > 200) continue;

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
        <section id="{section_id}" class="iteration-section">
            <div class="section-header">
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
            <div class="agent-grid">
                {agent_cards}
            </div>
        </section>
        '''

    def _generate_unity_section(self, unity_data: VisualizationData) -> str:
        """Generate the final unity section."""

        viz_data = json.dumps({
            "points": unity_data.points[:1000],
            "colors": unity_data.colors[:1000],
            "paths": unity_data.paths[:50],
            "arrows": unity_data.arrows[:50]
        })

        return f'''
        <section id="unity" class="unity-section">
            <h2>The Final Convergence</h2>
            <div class="formula">1 + 1 = 1</div>

            <div class="unity-canvas">
                <canvas id="unity-canvas" width="600" height="600"></canvas>
            </div>

            <p style="max-width: 600px; color: #888; margin-top: 2rem;">
                All agents - base and meta - collapse into a single point of understanding.
                The strange loop completes. Diversity becomes unity.
                The metaloop achieves what it sought: 1+1=1.
            </p>

            <script>
                (function() {{
                    const data = {viz_data};
                    const canvas = document.getElementById('unity-canvas');
                    const ctx = canvas.getContext('2d');
                    let frame = 0;

                    function draw() {{
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.03)';
                        ctx.fillRect(0, 0, 600, 600);
                        frame++;

                        const scale = 0.75;
                        const offset = 0;

                        // Draw paths
                        if (data.paths) {{
                            for (const path of data.paths) {{
                                if (path.length < 2) continue;
                                ctx.beginPath();
                                ctx.moveTo(path[0][0] * scale + offset, path[0][1] * scale + offset);
                                for (let i = 1; i < path.length; i++) {{
                                    ctx.lineTo(path[i][0] * scale + offset, path[i][1] * scale + offset);
                                }}
                                ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
                                ctx.lineWidth = 1.5;
                                ctx.stroke();
                            }}
                        }}

                        // Draw arrows
                        if (data.arrows) {{
                            for (const [start, end, color] of data.arrows) {{
                                ctx.beginPath();
                                ctx.moveTo(start[0] * scale + offset, start[1] * scale + offset);
                                ctx.lineTo(end[0] * scale + offset, end[1] * scale + offset);
                                ctx.strokeStyle = color || 'rgba(255, 215, 0, 0.5)';
                                ctx.lineWidth = 1;
                                ctx.stroke();
                            }}
                        }}

                        // Draw points with animation
                        if (data.points && data.colors) {{
                            for (let i = 0; i < data.points.length; i++) {{
                                const [x, y] = data.points[i];
                                const px = x * scale + offset;
                                const py = y * scale + offset;

                                const pulse = Math.sin(frame * 0.02 + i * 0.01) * 0.5 + 1;

                                ctx.beginPath();
                                ctx.arc(px, py, 2 * pulse, 0, Math.PI * 2);
                                ctx.fillStyle = data.colors[i] || '#ffd700';
                                ctx.fill();
                            }}
                        }}

                        // Central glow
                        const cx = 300, cy = 300;
                        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 100);
                        gradient.addColorStop(0, 'rgba(255, 215, 0, 0.3)');
                        gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
                        ctx.fillStyle = gradient;
                        ctx.beginPath();
                        ctx.arc(cx, cy, 100, 0, Math.PI * 2);
                        ctx.fill();

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
        </section>
        '''


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Run the METALOOP and generate the cathedral visualization.
    """
    print("\n" + "=" * 70)
    print("  METALOOP - Strange Loops in Mathematical Abstraction")
    print("  Hofstadter meets Grothendieck")
    print("=" * 70)

    # Create orchestrator
    orchestrator = MetaLoopOrchestrator(num_iterations=3)

    # Run the metaloop
    outputs = await orchestrator.run_metaloop()

    # Generate the cathedral HTML
    print("\nGenerating metaloop_cathedral.html...")
    html = orchestrator.generate_cathedral_html()

    output_path = "metaloop_cathedral.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Output written to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  METALOOP Complete")
    print("=" * 70)
    print(f"  Total iterations: {orchestrator.num_iterations}")
    print(f"  Total visualizations: {len(outputs)}")
    print(f"  Loop states recorded: {len(orchestrator.loop_states)}")
    print()
    print("  The strange loop has closed.")
    print("  Agents have observed agents observing agents.")
    print("  From many, one. 1+1=1.")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    asyncio.run(main())
