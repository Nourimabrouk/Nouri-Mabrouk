"""
NouriMabrouk Multi-Agent Swarm System - Grothendieck Edition
=============================================================

A swarm of 12 specialized agents exploring mathematics through Grothendieck's vision.
From the Nut & Sea metaphor through Category Theory, Topos, Schemes, and Motives,
to the Hermit's philosophical awakening and ultimate Synthesis.

Philosophy: 1+1=1 - Diverse agents converging into unity.

Original Agents:
- GrothendieckAgent: Rising sea abstraction
- EulerAgent: e^(iπ) + 1 = 0
- FibonacciAgent: Golden spiral
- MandelbrotAgent: Fractal complexity
- PrimeAgent: Ulam spiral

New Grothendieck-Inspired Agents:
- NutAndSeaAgent: Hammer vs Rising Sea paradigms
- CategoryTheoryAgent: Objects, morphisms, functors
- ToposAgent: Generalized spaces, sheaves, logic
- SchemeAgent: Algebraic geometry, Spec(R)
- MotivesAgent: Universal cohomology, 12 operations
- HermitAgent: Philosophical withdrawal, Récoltes et Semailles
- SynthesisAgent: 1+1=1 at the deepest level

Author: Nouri Mabrouk
"""

import asyncio
import math
import cmath
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import json
import colorsys
import random


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class Thought:
    """A unit of agent thinking - the quantum of cognition."""
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class Message:
    """Inter-agent communication unit."""
    sender: str
    receiver: str  # "all" for broadcast
    thought: Thought
    message_type: str = "contribution"  # contribution, request, response, insight


@dataclass
class VisualizationData:
    """Data for rendering mathematical beauty."""
    points: list[tuple[float, float]] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    paths: list[list[tuple[float, float]]] = field(default_factory=list)
    arrows: list[tuple[tuple[float, float], tuple[float, float], str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    html_fragment: str = ""  # Custom HTML/SVG for complex visualizations


@dataclass
class CrossReference:
    """Reference between agent visualizations."""
    source_agent: str
    target_agent: str
    connection_type: str
    description: str


# ============================================================================
# Message Bus - The Medium of Unity
# ============================================================================

class MessageBus:
    """
    Shared communication channel for the swarm.
    Like the ether through which mathematical truths propagate.
    """

    def __init__(self):
        self._messages: asyncio.Queue[Message] = asyncio.Queue()
        self._subscribers: dict[str, asyncio.Queue[Message]] = {}
        self._history: list[Message] = []
        self._insights: dict[str, list[Thought]] = {}
        self._running = False

    def subscribe(self, agent_id: str) -> asyncio.Queue[Message]:
        """Subscribe an agent to receive messages."""
        queue: asyncio.Queue[Message] = asyncio.Queue()
        self._subscribers[agent_id] = queue
        self._insights[agent_id] = []
        return queue

    async def publish(self, message: Message) -> None:
        """Publish a message to the bus."""
        self._history.append(message)
        if message.message_type == "insight":
            self._insights[message.sender].append(message.thought)
        await self._messages.put(message)

    async def run(self) -> None:
        """Route messages to subscribers."""
        self._running = True
        while self._running:
            try:
                message = await asyncio.wait_for(self._messages.get(), timeout=0.1)
                if message.receiver == "all":
                    for agent_id, queue in self._subscribers.items():
                        if agent_id != message.sender:
                            await queue.put(message)
                elif message.receiver in self._subscribers:
                    await self._subscribers[message.receiver].put(message)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the message bus."""
        self._running = False

    def get_all_insights(self) -> dict[str, list[Thought]]:
        """Get all insights from all agents."""
        return self._insights

    def get_cross_references(self) -> list[CrossReference]:
        """Analyze message history for cross-references."""
        refs = []
        for msg in self._history:
            if msg.message_type == "insight" and msg.receiver == "all":
                # Create potential cross-references based on message content
                for other_agent in self._subscribers:
                    if other_agent != msg.sender:
                        refs.append(CrossReference(
                            source_agent=msg.sender,
                            target_agent=other_agent,
                            connection_type="insight_sharing",
                            description=msg.thought.content[:50]
                        ))
        return refs


# ============================================================================
# Base Agent - The Platonic Form
# ============================================================================

class MathematicalAgent(ABC):
    """
    Base class for all mathematical agents.
    Implements the think-act-observe loop.
    """

    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus = bus
        self.inbox = bus.subscribe(agent_id)
        self.observations: list[Thought] = []
        self.contributions: list[VisualizationData] = []

    @abstractmethod
    async def think(self, context: dict[str, Any]) -> Thought:
        """Generate a thought based on context."""
        pass

    @abstractmethod
    async def act(self, thought: Thought) -> VisualizationData:
        """Transform thought into visualization data."""
        pass

    async def observe(self, data: VisualizationData) -> None:
        """Integrate observations from action results."""
        self.contributions.append(data)

    async def broadcast(self, thought: Thought) -> None:
        """Share a thought with all agents."""
        message = Message(
            sender=self.agent_id,
            receiver="all",
            thought=thought
        )
        await self.bus.publish(message)

    async def share_insight(self, insight: str, data: dict[str, Any] = None) -> None:
        """Share an insight with all agents."""
        thought = Thought(
            content=insight,
            data=data or {},
            agent_id=self.agent_id
        )
        message = Message(
            sender=self.agent_id,
            receiver="all",
            thought=thought,
            message_type="insight"
        )
        await self.bus.publish(message)

    async def receive(self, timeout: float = 0.5) -> Message | None:
        """Receive a message if available."""
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def run_cycle(self, context: dict[str, Any]) -> VisualizationData:
        """Execute one think-act-observe cycle."""
        thought = await self.think(context)
        await self.broadcast(thought)
        data = await self.act(thought)
        await self.observe(data)
        return data


# ============================================================================
# Original Agents - The Foundation
# ============================================================================

class GrothendieckAgent(MathematicalAgent):
    """
    Raises the sea to sink the nut.
    Finds unifying structures, builds abstraction layers.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("grothendieck", bus)
        self.abstraction_level = 0

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the rising sea of abstraction."""
        self.abstraction_level += 1
        return Thought(
            content="The sea rises. Structures that seemed separate are revealed as one.",
            data={
                "abstraction_level": self.abstraction_level,
                "unifying_principle": "topos",
                "metaphor": "rising_sea"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create visualization of rising abstraction levels."""
        points = []
        colors = []
        paths = []

        num_layers = 7
        points_per_layer = 60

        for layer in range(num_layers):
            radius = 50 + layer * 40
            alpha = 0.3 + (layer / num_layers) * 0.5

            layer_points = []
            for i in range(points_per_layer):
                angle = 2 * math.pi * i / points_per_layer
                wave = math.sin(angle * 3 + layer) * 5
                x = 400 + (radius + wave) * math.cos(angle)
                y = 400 + (radius + wave) * math.sin(angle)
                layer_points.append((x, y))
                points.append((x, y))

                hue = 0.6 - layer * 0.02
                sat = 0.8 - layer * 0.05
                val = 0.5 + layer * 0.07
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})")

            paths.append(layer_points + [layer_points[0]])

        await self.share_insight(
            "Abstraction is not escape from reality, but deeper engagement with it.",
            {"layers": num_layers}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            paths=paths,
            metadata={"agent": "grothendieck", "theme": "rising_sea"}
        )


class EulerAgent(MathematicalAgent):
    """
    Guardian of e^(ipi) + 1 = 0.
    The most beautiful equation - five fundamental constants in perfect unity.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("euler", bus)
        self.constants = {"e": math.e, "i": 1j, "pi": math.pi, "one": 1, "zero": 0}

    async def think(self, context: dict[str, Any]) -> Thought:
        """Meditate on e^(ipi) + 1 = 0."""
        result = cmath.exp(1j * math.pi) + 1
        return Thought(
            content=f"e^(ipi) + 1 = {result.real:.2e}. Five constants, one truth.",
            data={
                "constants": {k: str(v) for k, v in self.constants.items()},
                "identity_verified": abs(result) < 1e-15,
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Visualize the five constants and their unity."""
        points = []
        colors = []
        labels = []
        paths = []

        center_x, center_y = 400, 400
        constant_names = ["e", "i", "pi", "1", "0"]
        constant_colors = ["#e74c3c", "#9b59b6", "#3498db", "#2ecc71", "#34495e"]

        for idx, (name, color) in enumerate(zip(constant_names, constant_colors)):
            angle = -math.pi/2 + (2 * math.pi * idx / 5)
            radius = 150
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
            colors.append(color)
            labels.append(name)

        for i in range(5):
            for j in range(i + 1, 5):
                paths.append([points[i], points[j]])

        circle_points = []
        for i in range(101):
            angle = 2 * math.pi * i / 100
            x = center_x + 100 * math.cos(angle)
            y = center_y + 100 * math.sin(angle)
            circle_points.append((x, y))
        paths.append(circle_points)

        euler_x = center_x - 100
        euler_y = center_y
        points.append((euler_x, euler_y))
        colors.append("#f39c12")
        labels.append("e^(ipi)")

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            metadata={"agent": "euler", "theme": "five_constants"}
        )


class FibonacciAgent(MathematicalAgent):
    """Keeper of the golden ratio phi = (1 + sqrt(5)) / 2."""

    def __init__(self, bus: MessageBus):
        super().__init__("fibonacci", bus)
        self.phi = (1 + math.sqrt(5)) / 2
        self.sequence = [0, 1]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the golden spiral."""
        while len(self.sequence) < 20:
            self.sequence.append(self.sequence[-1] + self.sequence[-2])
        return Thought(
            content=f"phi = {self.phi:.10f}. Each number is the sum of the two before.",
            data={"phi": self.phi, "sequence": self.sequence[:15]},
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create the golden spiral visualization."""
        points = []
        colors = []
        paths = []
        center_x, center_y = 400, 400

        spiral_points = []
        a = 5
        b = math.log(self.phi) / (math.pi / 2)

        for i in range(500):
            theta = i * 0.1
            r = a * math.exp(b * theta)
            if r > 300:
                break
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            spiral_points.append((x, y))
            points.append((x, y))
            t = i / 500
            hue = 0.1 + t * 0.2
            r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f"rgb({int(r_c*255)}, {int(g_c*255)}, {int(b_c*255)})")

        paths.append(spiral_points)

        return VisualizationData(
            points=points,
            colors=colors,
            paths=paths,
            metadata={"agent": "fibonacci", "theme": "golden_spiral", "phi": self.phi}
        )


class MandelbrotAgent(MathematicalAgent):
    """Explorer of infinite complexity from simplicity: z^2 + c."""

    def __init__(self, bus: MessageBus):
        super().__init__("mandelbrot", bus)
        self.max_iter = 100

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate infinite complexity from z^2 + c."""
        return Thought(
            content="z^2 + c. Two operations, infinite complexity.",
            data={"formula": "z_{n+1} = z_n^2 + c", "complexity": "infinite"},
            agent_id=self.agent_id
        )

    def _mandelbrot_escape(self, c: complex, max_iter: int) -> int:
        """Calculate escape time for a point."""
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z * z + c
        return max_iter

    async def act(self, thought: Thought) -> VisualizationData:
        """Generate Mandelbrot set visualization data."""
        points = []
        colors = []
        center_x, center_y = 400, 400
        scale = 200

        for px in range(0, 800, 4):
            for py in range(0, 800, 4):
                real = (px - center_x) / scale - 0.5
                imag = (py - center_y) / scale
                c = complex(real, imag)
                escape = self._mandelbrot_escape(c, self.max_iter)

                if escape < self.max_iter:
                    points.append((px, py))
                    t = escape / self.max_iter
                    hue = 0.7 - t * 0.5
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.3 + t * 0.7)
                    colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")

        return VisualizationData(
            points=points,
            colors=colors,
            metadata={"agent": "mandelbrot", "theme": "fractal_boundary"}
        )


class PrimeAgent(MathematicalAgent):
    """Seeker of primes, the atoms of arithmetic."""

    def __init__(self, bus: MessageBus):
        super().__init__("prime", bus)
        self.primes = self._sieve(1000)

    def _sieve(self, n: int) -> list[int]:
        """Sieve of Eratosthenes."""
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(n + 1) if is_prime[i]]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the distribution of primes."""
        twins = [(p, p+2) for p in self.primes if p+2 in self.primes]
        return Thought(
            content=f"Found {len(self.primes)} primes, including {len(twins)} twin pairs.",
            data={"count": len(self.primes), "largest": self.primes[-1], "twin_primes": len(twins)},
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create Ulam spiral visualization."""
        points = []
        colors = []
        center_x, center_y = 400, 400
        scale = 4

        x, y = 0, 0
        dx, dy = 1, 0
        steps_in_direction = 1
        steps_taken = 0
        direction_changes = 0
        prime_set = set(self.primes)

        for n in range(1, 2501):
            if n in prime_set:
                px = center_x + x * scale
                py = center_y + y * scale
                points.append((px, py))
                t = n / 2500
                hue = 0.0 + t * 0.15
                r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
                colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")

            x += dx
            y += dy
            steps_taken += 1

            if steps_taken == steps_in_direction:
                steps_taken = 0
                dx, dy = -dy, dx
                direction_changes += 1
                if direction_changes % 2 == 0:
                    steps_in_direction += 1

        return VisualizationData(
            points=points,
            colors=colors,
            metadata={"agent": "prime", "theme": "ulam_spiral"}
        )


# ============================================================================
# New Grothendieck-Inspired Agents
# ============================================================================

class NutAndSeaAgent(MathematicalAgent):
    """
    Visualizes Grothendieck's two approaches:
    - The Hammer: attacking the nut directly, brute force
    - The Rising Sea: raising abstraction until the problem dissolves
    """

    def __init__(self, bus: MessageBus):
        super().__init__("nut_and_sea", bus)
        self.paradigm = "sea"  # or "hammer"
        self.water_level = 0.0

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the two paradigms of problem-solving."""
        self.water_level += 0.1
        return Thought(
            content="The hammer cracks, but the sea dissolves. "
                    "Brute force yields to patient abstraction.",
            data={
                "hammer_approach": "direct attack on specific problem",
                "sea_approach": "build general theory until problem becomes trivial",
                "water_level": self.water_level,
                "grothendieck_quote": "Je n'aime pas frapper sur un clou avec un marteau"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create dual visualization: hammer vs rising sea."""
        points = []
        colors = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # The Nut (problem) - represented as a hard polygon
        nut_radius = 60
        nut_sides = 8
        nut_points = []
        for i in range(nut_sides):
            angle = 2 * math.pi * i / nut_sides - math.pi / nut_sides
            x = center_x + nut_radius * math.cos(angle)
            y = center_y - 50 + nut_radius * math.sin(angle)
            nut_points.append((x, y))
            points.append((x, y))
            colors.append("#8B4513")  # Brown for nut
        nut_points.append(nut_points[0])  # Close the shape
        paths.append(nut_points)

        # The Hammer (on left side) - brute force approach
        hammer_x = center_x - 200
        hammer_y = center_y - 100

        # Hammer head
        hammer_head = [
            (hammer_x - 30, hammer_y - 20),
            (hammer_x + 30, hammer_y - 20),
            (hammer_x + 30, hammer_y + 20),
            (hammer_x - 30, hammer_y + 20),
            (hammer_x - 30, hammer_y - 20)
        ]
        paths.append(hammer_head)
        for p in hammer_head[:-1]:
            points.append(p)
            colors.append("#555555")

        # Hammer handle
        handle = [
            (hammer_x, hammer_y + 20),
            (hammer_x, hammer_y + 100)
        ]
        paths.append(handle)

        # Force arrows pointing at nut
        arrows.append(((hammer_x + 50, hammer_y), (center_x - 80, center_y - 50), "#ff4444"))
        arrows.append(((hammer_x + 50, hammer_y + 30), (center_x - 80, center_y - 30), "#ff6666"))

        # The Rising Sea (on right side and below)
        water_level = thought.data["water_level"]
        num_waves = 5

        for wave_idx in range(num_waves):
            wave_points = []
            wave_y = center_y + 150 - wave_idx * 30 * water_level
            amplitude = 15 + wave_idx * 5

            for i in range(100):
                x = 150 + i * 5
                phase = wave_idx * 0.5 + i * 0.1
                y = wave_y + amplitude * math.sin(phase)
                wave_points.append((x, y))
                points.append((x, y))

                # Gradient from deep blue to light
                alpha = 0.3 + wave_idx * 0.1
                r, g, b = colorsys.hsv_to_rgb(0.55 + wave_idx * 0.02, 0.7, 0.5 + wave_idx * 0.1)
                colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})")

            paths.append(wave_points)

        # Rising arrows showing water level increasing
        arrows.append(((center_x + 150, center_y + 100), (center_x + 150, center_y - 20), "#4488ff"))
        arrows.append(((center_x + 180, center_y + 80), (center_x + 180, center_y), "#66aaff"))

        await self.share_insight(
            "The sea rises not by force, but by patient accumulation of understanding.",
            {"water_level": water_level}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            paths=paths,
            arrows=arrows,
            labels=["Hammer", "Nut", "Rising Sea"],
            metadata={
                "agent": "nut_and_sea",
                "theme": "paradigms",
                "water_level": water_level
            }
        )


class CategoryTheoryAgent(MathematicalAgent):
    """
    Objects and morphisms: the language of mathematical structure.
    Functors, natural transformations, universal properties.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("category_theory", bus)
        self.objects = ["A", "B", "C", "D", "E"]
        self.morphisms = [
            ("A", "B", "f"), ("B", "C", "g"), ("A", "C", "g o f"),
            ("C", "D", "h"), ("D", "E", "k"), ("C", "E", "k o h"),
            ("B", "D", "j")
        ]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the arrow-theoretic view of mathematics."""
        return Thought(
            content="Objects are points, morphisms are arrows. "
                    "What matters is not what things ARE, but how they RELATE.",
            data={
                "objects": self.objects,
                "morphisms": [(s, t, n) for s, t, n in self.morphisms],
                "principles": [
                    "Composition is associative",
                    "Every object has identity morphism",
                    "Functors preserve structure",
                    "Natural transformations are morphisms between functors"
                ]
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create category diagram visualization."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Position objects in a pentagon/circle
        object_positions = {}
        num_objects = len(self.objects)

        for idx, obj in enumerate(self.objects):
            angle = -math.pi/2 + 2 * math.pi * idx / num_objects
            radius = 180
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            object_positions[obj] = (x, y)

            # Object as a filled circle
            for theta in range(0, 360, 30):
                rad = math.radians(theta)
                px = x + 20 * math.cos(rad)
                py = y + 20 * math.sin(rad)
                points.append((px, py))
                colors.append("#4a90d9")

            points.append((x, y))
            colors.append("#2c5282")
            labels.append(obj)

        # Draw morphisms as arrows
        for source, target, name in self.morphisms:
            sx, sy = object_positions[source]
            tx, ty = object_positions[target]

            # Offset start and end to not overlap with object circles
            dx, dy = tx - sx, ty - sy
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx/length, dy/length

            start = (sx + dx * 25, sy + dy * 25)
            end = (tx - dx * 25, ty - dy * 25)

            # Curved path for morphisms (bezier approximation)
            if name in ["g o f", "k o h"]:  # Composition - draw curved
                mid_x = (start[0] + end[0]) / 2 + 30
                mid_y = (start[1] + end[1]) / 2 - 30
                path_points = [start, (mid_x, mid_y), end]
                paths.append(path_points)
                arrows.append((start, end, "#e74c3c"))  # Red for composition
            else:
                arrows.append((start, end, "#48bb78"))  # Green for basic morphisms

            # Label position
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 - 15
            points.append((mid_x, mid_y))
            colors.append("#888888")
            labels.append(name)

        # Draw a functor arrow between two "categories" (dotted box regions)
        # Left category region
        left_box = [(200, 200), (350, 200), (350, 450), (200, 450), (200, 200)]
        paths.append(left_box)

        # Right category region
        right_box = [(450, 200), (600, 200), (600, 450), (450, 450), (450, 200)]
        paths.append(right_box)

        # Functor arrow between categories
        arrows.append(((360, 325), (440, 325), "#9b59b6"))  # Purple for functor

        await self.share_insight(
            "Category theory reveals: structure is preserved through morphisms, not encoded in objects.",
            {"objects": len(self.objects), "morphisms": len(self.morphisms)}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "category_theory",
                "theme": "objects_and_morphisms"
            }
        )


class ToposAgent(MathematicalAgent):
    """
    Generalized spaces: where logic meets geometry.
    Sheaves, subobject classifiers, internal logic.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("topos", bus)
        self.sheaf_sections = {}

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the unity of logic and geometry in topos theory."""
        return Thought(
            content="A topos is a universe. It has its own logic, its own truth values, "
                    "its own notion of space. Sets are just one example.",
            data={
                "key_concepts": [
                    "Sheaves: local data that patches together globally",
                    "Subobject classifier Omega: generalizes {true, false}",
                    "Internal logic: every topos has its own truth values",
                    "Geometric morphisms: structure-preserving maps between universes"
                ],
                "examples": ["Set", "Sh(X) - sheaves on space X", "BG - G-sets"],
                "philosophy": "Different topoi, different mathematics"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create visualization of topos concepts."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Base topological space (a simple topology on 3 points)
        # Represented as nested ovals (open sets)
        space_colors = ["#3498db", "#2ecc71", "#e74c3c"]

        # Three "open sets" as ovals
        for idx, (cx_offset, cy_offset, rx, ry) in enumerate([
            (0, 0, 200, 150),       # Whole space
            (-50, 20, 100, 80),     # Smaller open set
            (50, -20, 80, 60)       # Another open set
        ]):
            oval_points = []
            for i in range(50):
                angle = 2 * math.pi * i / 50
                x = center_x + cx_offset + rx * math.cos(angle)
                y = center_y + cy_offset + ry * math.sin(angle)
                oval_points.append((x, y))
                points.append((x, y))
                colors.append(space_colors[idx])
            oval_points.append(oval_points[0])
            paths.append(oval_points)

        # Sheaf sections over open sets (represented as "data" above each open set)
        # Section F(U) - a fiber/stalk above each open set
        section_data = [
            (center_x, center_y - 200, "F(X)", "Global section"),
            (center_x - 100, center_y - 150, "F(U)", "Local section"),
            (center_x + 100, center_y - 150, "F(V)", "Local section"),
        ]

        for sx, sy, label, desc in section_data:
            points.append((sx, sy))
            colors.append("#9b59b6")
            labels.append(label)

            # Arrow from section to open set
            arrows.append(((sx, sy + 15), (sx, center_y - 80), "#9b59b6"))

        # Restriction maps between sections
        arrows.append(((center_x - 30, center_y - 200), (center_x - 90, center_y - 160), "#ff9500"))
        arrows.append(((center_x + 30, center_y - 200), (center_x + 90, center_y - 160), "#ff9500"))

        # Subobject classifier Omega (truth value object)
        omega_x, omega_y = center_x + 250, center_y
        omega_points = []
        for i in range(30):
            angle = 2 * math.pi * i / 30
            x = omega_x + 40 * math.cos(angle)
            y = omega_y + 40 * math.sin(angle)
            omega_points.append((x, y))
            points.append((x, y))
            colors.append("#f1c40f")  # Gold for omega
        omega_points.append(omega_points[0])
        paths.append(omega_points)

        points.append((omega_x, omega_y))
        colors.append("#f39c12")
        labels.append("Omega")

        # Internal logic: true/false inside Omega
        points.append((omega_x - 15, omega_y))
        colors.append("#27ae60")
        labels.append("T")

        points.append((omega_x + 15, omega_y))
        colors.append("#c0392b")
        labels.append("...")  # Many truth values possible

        await self.share_insight(
            "In a topos, truth is not binary. The subobject classifier Omega contains all possible truth values.",
            {"concept": "internal_logic"}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "topos",
                "theme": "generalized_spaces"
            }
        )


class SchemeAgent(MathematicalAgent):
    """
    Algebraic geometry revolutionized: Spec(R), generic points, Zariski topology.
    Where algebra and geometry become one.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("scheme", bus)

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the scheme-theoretic view of geometry."""
        return Thought(
            content="A scheme is algebra made geometric. Every ring has a spectrum, "
                    "every ideal a geometric locus. Nilpotents are infinitesimal fuzz.",
            data={
                "key_concepts": [
                    "Spec(R): prime ideals of R form a space",
                    "Generic point: the zero ideal, contains all",
                    "Zariski topology: closed sets = vanishing loci",
                    "Structure sheaf: algebraic data on geometric space",
                    "Nilpotents: remember infinitesimal directions"
                ],
                "examples": [
                    "Spec(Z): the 'space' of prime numbers",
                    "Spec(k[x]): affine line",
                    "Spec(k[x,y]/(xy)): two lines meeting"
                ]
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create visualization of scheme concepts."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Spec(Z) visualization: prime points (2), (3), (5), (7), ... and generic point (0)

        # Generic point (0) at center - the "big" point
        generic_points = []
        for i in range(40):
            angle = 2 * math.pi * i / 40
            x = center_x + 30 * math.cos(angle)
            y = center_y - 100 + 30 * math.sin(angle)
            generic_points.append((x, y))
            points.append((x, y))
            colors.append("#f1c40f")  # Gold for generic point
        generic_points.append(generic_points[0])
        paths.append(generic_points)
        points.append((center_x, center_y - 100))
        colors.append("#f39c12")
        labels.append("(0)")

        # Prime points arranged in a line below, connected to generic
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for idx, p in enumerate(primes):
            px = center_x - 200 + idx * 57
            py = center_y + 50

            # Small circle for prime point
            for i in range(20):
                angle = 2 * math.pi * i / 20
                x = px + 15 * math.cos(angle)
                y = py + 15 * math.sin(angle)
                points.append((x, y))
                hue = 0.6 - (idx / len(primes)) * 0.3
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")

            points.append((px, py))
            colors.append("#2980b9")
            labels.append(f"({p})")

            # Specialization arrow: generic -> prime
            arrows.append(((center_x, center_y - 65), (px, py - 20), "#888888"))

        # Affine line Spec(k[x]) on the right
        line_start_x = center_x + 100
        line_start_y = center_y + 150
        line_points = []

        for i in range(100):
            x = line_start_x + i * 2
            y = line_start_y
            line_points.append((x, y))
            points.append((x, y))
            colors.append("#2ecc71")

        paths.append(line_points)
        labels.append("Spec(k[x])")
        points.append((line_start_x + 100, line_start_y - 15))
        colors.append("#27ae60")

        # Generic point of affine line
        points.append((line_start_x + 100, line_start_y))
        colors.append("#f39c12")
        labels.append("eta")

        # Nilpotent "fuzz" - small fuzzy region to show infinitesimal structure
        fuzz_x, fuzz_y = center_x - 200, center_y + 150
        for _ in range(50):
            x = fuzz_x + random.gauss(0, 15)
            y = fuzz_y + random.gauss(0, 15)
            points.append((x, y))
            colors.append("rgba(155, 89, 182, 0.5)")

        points.append((fuzz_x, fuzz_y - 30))
        colors.append("#9b59b6")
        labels.append("Nilpotent fuzz")

        await self.share_insight(
            "Schemes unify number theory and geometry: Spec(Z) is the 'base' of all schemes over integers.",
            {"primes_shown": len(primes)}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "scheme",
                "theme": "algebraic_geometry"
            }
        )


class MotivesAgent(MathematicalAgent):
    """
    The universal cohomology theory: motives.
    12 operations, Weil conjectures, the yoga of weights.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("motives", bus)
        self.cohomology_theories = [
            "Singular", "de Rham", "Etale", "Crystalline",
            "l-adic", "p-adic", "Hodge", "Motivic"
        ]
        self.twelve_motives = [
            "Tate(1)", "h^0", "h^1", "h^2",
            "Lefschetz", "Artin", "Chow", "Voevodsky",
            "K-theory", "Algebraic", "Mixed", "Pure"
        ]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the universal nature of motives."""
        return Thought(
            content="A motive is the 'soul' of a variety - what remains when all "
                    "cohomology theories converge. The yoga of motives reveals deep unity.",
            data={
                "cohomology_theories": self.cohomology_theories,
                "weil_conjectures": [
                    "Rationality", "Functional equation",
                    "Riemann hypothesis (for varieties)", "Betti numbers"
                ],
                "six_operations": [
                    "f^* (pullback)", "f_* (pushforward)",
                    "f^! (exceptional pullback)", "f_! (proper pushforward)",
                    "tensor", "internal Hom"
                ],
                "philosophy": "All roads lead to motives"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create visualization of motive concepts - the 12 motives converging."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Central "universal motive" - the convergence point
        central_points = []
        for i in range(50):
            angle = 2 * math.pi * i / 50
            x = center_x + 50 * math.cos(angle)
            y = center_y + 50 * math.sin(angle)
            central_points.append((x, y))
            points.append((x, y))
            colors.append("#f1c40f")  # Gold
        central_points.append(central_points[0])
        paths.append(central_points)

        points.append((center_x, center_y))
        colors.append("#f39c12")
        labels.append("M(X)")

        # 12 motives/cohomology theories arranged in a circle
        outer_radius = 200
        theory_colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
            "#f39c12", "#1abc9c", "#e67e22", "#34495e",
            "#16a085", "#c0392b", "#8e44ad", "#27ae60"
        ]

        for idx, (motive, color) in enumerate(zip(self.twelve_motives, theory_colors)):
            angle = 2 * math.pi * idx / 12 - math.pi / 2
            x = center_x + outer_radius * math.cos(angle)
            y = center_y + outer_radius * math.sin(angle)

            # Node for each motive
            node_points = []
            for i in range(20):
                node_angle = 2 * math.pi * i / 20
                nx = x + 25 * math.cos(node_angle)
                ny = y + 25 * math.sin(node_angle)
                node_points.append((nx, ny))
                points.append((nx, ny))
                colors.append(color)
            node_points.append(node_points[0])
            paths.append(node_points)

            points.append((x, y))
            colors.append(color)
            labels.append(motive)

            # Arrow from motive to center (convergence)
            dx = center_x - x
            dy = center_y - y
            length = math.sqrt(dx*dx + dy*dy)
            dx, dy = dx/length, dy/length

            start = (x + dx * 30, y + dy * 30)
            end = (center_x - dx * 55, center_y - dy * 55)
            arrows.append((start, end, color))

        # Cohomology theories as labels around the edge
        text_radius = 280
        for idx, theory in enumerate(self.cohomology_theories):
            angle = 2 * math.pi * idx / len(self.cohomology_theories)
            x = center_x + text_radius * math.cos(angle)
            y = center_y + text_radius * math.sin(angle)
            points.append((x, y))
            colors.append("#888888")
            labels.append(theory)

        # Draw "six operations" as curved arrows between adjacent motives
        for i in range(0, 12, 2):
            angle1 = 2 * math.pi * i / 12 - math.pi / 2
            angle2 = 2 * math.pi * ((i + 1) % 12) / 12 - math.pi / 2

            x1 = center_x + outer_radius * math.cos(angle1)
            y1 = center_y + outer_radius * math.sin(angle1)
            x2 = center_x + outer_radius * math.cos(angle2)
            y2 = center_y + outer_radius * math.sin(angle2)

            # Curved path
            mid_angle = (angle1 + angle2) / 2
            mid_radius = outer_radius + 40
            mx = center_x + mid_radius * math.cos(mid_angle)
            my = center_y + mid_radius * math.sin(mid_angle)

            paths.append([(x1, y1), (mx, my), (x2, y2)])

        await self.share_insight(
            "Motives are the universal cohomology theory - all others factor through them.",
            {"cohomologies": len(self.cohomology_theories), "operations": 6}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "motives",
                "theme": "universal_cohomology",
                "twelve_motives": self.twelve_motives
            }
        )


class HermitAgent(MathematicalAgent):
    """
    The philosophical awakening: Grothendieck's withdrawal.
    Recoltes et Semailles, Survivre et Vivre, ecological consciousness.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("hermit", bus)
        self.meditations = [
            "The mathematical community has lost its way",
            "Truth cannot be owned or stolen",
            "Return to the source, to direct experience",
            "Mathematics emerged from meditation, not competition",
            "The violence of academic structures mirrors society"
        ]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the hermit's withdrawal and awakening."""
        meditation = random.choice(self.meditations)
        return Thought(
            content=f"From Recoltes et Semailles: '{meditation}'",
            data={
                "works": [
                    "Recoltes et Semailles (Harvests and Sowings)",
                    "La Clef des Songes (The Key to Dreams)",
                    "Survivre et Vivre (To Survive and Live)"
                ],
                "themes": [
                    "Critique of mathematical establishment",
                    "Ecological consciousness",
                    "Spiritual awakening",
                    "Return to simplicity",
                    "The violence of abstraction without wisdom"
                ],
                "final_years": "1991-2014, hermitage in Lasserre, Pyrenees"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create meditative visualization - geometric patterns of withdrawal."""
        points = []
        colors = []
        labels = []
        paths = []

        center_x, center_y = 400, 400

        # Spiral of withdrawal - going inward
        spiral_points = []
        for i in range(300):
            t = i / 300
            angle = t * 6 * math.pi  # 3 full turns
            radius = 250 * (1 - t * 0.8)  # Shrinking radius - going inward

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            spiral_points.append((x, y))
            points.append((x, y))

            # Colors fade from active (bright) to contemplative (muted)
            hue = 0.7 - t * 0.3  # Blue to purple
            sat = 0.8 - t * 0.4  # Desaturating
            val = 0.9 - t * 0.3  # Dimming
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")

        paths.append(spiral_points)

        # Center point - the hermitage
        for i in range(30):
            angle = 2 * math.pi * i / 30
            x = center_x + 30 * math.cos(angle)
            y = center_y + 30 * math.sin(angle)
            points.append((x, y))
            colors.append("#2c3e50")  # Dark blue-gray

        points.append((center_x, center_y))
        colors.append("#1a252f")
        labels.append("Lasserre")

        # Mountain silhouette (Pyrenees) at bottom
        mountain_points = [(200, 600)]
        peaks = [(250, 520), (320, 480), (400, 450), (480, 470), (550, 490), (600, 600)]
        for peak in peaks:
            mountain_points.append(peak)
            points.append(peak)
            colors.append("#34495e")
        mountain_points.append((600, 600))
        paths.append(mountain_points)

        # Stars representing dreams/visions
        for _ in range(50):
            x = random.uniform(150, 650)
            y = random.uniform(100, 350)
            if random.random() > 0.7:
                points.append((x, y))
                colors.append("#f1c40f" if random.random() > 0.5 else "#ecf0f1")

        # Sacred geometry pattern - overlapping circles (flower of life hint)
        sacred_radius = 40
        for ring in range(3):
            ring_radius = 80 + ring * 50
            num_circles = 6 * (ring + 1)
            for i in range(num_circles):
                angle = 2 * math.pi * i / num_circles + ring * 0.2
                cx = center_x + ring_radius * math.cos(angle)
                cy = center_y + ring_radius * math.sin(angle)

                circle_pts = []
                for j in range(25):
                    ca = 2 * math.pi * j / 25
                    px = cx + sacred_radius * math.cos(ca)
                    py = cy + sacred_radius * math.sin(ca)
                    circle_pts.append((px, py))
                    points.append((px, py))
                    alpha = 0.1 + ring * 0.05
                    colors.append(f"rgba(200, 200, 220, {alpha})")
                circle_pts.append(circle_pts[0])
                paths.append(circle_pts)

        # Key quotes as labels
        quotes = [
            ("The Dreaming", (center_x - 250, center_y - 200)),
            ("Survivre", (center_x + 200, center_y - 180)),
            ("et Vivre", (center_x + 200, center_y - 160)),
        ]
        for quote, (qx, qy) in quotes:
            points.append((qx, qy))
            colors.append("#95a5a6")
            labels.append(quote)

        await self.share_insight(
            "The deepest mathematics comes not from ambition, but from patient listening to truth.",
            {"location": "Lasserre", "years": "1991-2014"}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            metadata={
                "agent": "hermit",
                "theme": "philosophical_withdrawal",
                "meditation": thought.content
            }
        )


class SynthesisAgent(MathematicalAgent):
    """
    1+1=1 at the deepest level: ultimate synthesis.
    Cosmic Galois group, anabelian geometry, the unity of mathematics.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("synthesis", bus)
        self.previous_agents = [
            "nut_and_sea", "category_theory", "topos",
            "scheme", "motives", "hermit"
        ]

    async def think(self, context: dict[str, Any]) -> Thought:
        """Contemplate the ultimate synthesis - 1+1=1."""
        return Thought(
            content="All mathematics is one. Category theory, topos theory, schemes, motives - "
                    "they are facets of a single diamond. The cosmic Galois group acts on everything.",
            data={
                "unity_principles": [
                    "1+1=1: Diversity converges to unity",
                    "Cosmic Galois group: fundamental group of arithmetic",
                    "Anabelian geometry: spaces from their fundamental groups",
                    "The motivic yoga: all cohomologies are one",
                    "Rising sea: abstraction reveals unity"
                ],
                "grothendieck_vision": "A mathematics of natural structures, "
                                       "where everything connects",
                "final_synthesis": "The heart of mathematics beats as one"
            },
            agent_id=self.agent_id
        )

    async def act(self, thought: Thought) -> VisualizationData:
        """Create the ultimate synthesis visualization."""
        points = []
        colors = []
        labels = []
        paths = []
        arrows = []

        center_x, center_y = 400, 400

        # Central unity point - pulsing, radiant
        for radius in range(10, 60, 5):
            alpha = 1.0 - radius / 60
            for i in range(50):
                angle = 2 * math.pi * i / 50
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
                r, g, b = colorsys.hsv_to_rgb(0.1, 0.3, 1.0)  # Warm white/gold
                colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})")

        points.append((center_x, center_y))
        colors.append("#ffffff")
        labels.append("1+1=1")

        # Previous agents as nodes converging to center
        agent_colors = {
            "nut_and_sea": "#3498db",
            "category_theory": "#9b59b6",
            "topos": "#2ecc71",
            "scheme": "#e74c3c",
            "motives": "#f39c12",
            "hermit": "#1abc9c"
        }

        outer_radius = 280
        for idx, agent in enumerate(self.previous_agents):
            angle = 2 * math.pi * idx / len(self.previous_agents) - math.pi / 2
            ax = center_x + outer_radius * math.cos(angle)
            ay = center_y + outer_radius * math.sin(angle)

            # Agent node
            node_color = agent_colors.get(agent, "#888888")
            for i in range(25):
                node_angle = 2 * math.pi * i / 25
                x = ax + 35 * math.cos(node_angle)
                y = ay + 35 * math.sin(node_angle)
                points.append((x, y))
                colors.append(node_color)

            points.append((ax, ay))
            colors.append(node_color)
            labels.append(agent.replace("_", " ").title())

            # Convergence arrow to center
            dx = center_x - ax
            dy = center_y - ay
            length = math.sqrt(dx*dx + dy*dy)
            dx, dy = dx/length, dy/length

            start = (ax + dx * 40, ay + dy * 40)
            end = (center_x - dx * 65, center_y - dy * 65)
            arrows.append((start, end, node_color))

            # Connections between adjacent agents (the web of mathematics)
            next_idx = (idx + 1) % len(self.previous_agents)
            next_angle = 2 * math.pi * next_idx / len(self.previous_agents) - math.pi / 2
            next_x = center_x + outer_radius * math.cos(next_angle)
            next_y = center_y + outer_radius * math.sin(next_angle)

            # Curved connection
            mid_angle = (angle + next_angle) / 2
            mid_radius = outer_radius + 30
            mx = center_x + mid_radius * math.cos(mid_angle)
            my = center_y + mid_radius * math.sin(mid_angle)
            paths.append([(ax, ay), (mx, my), (next_x, next_y)])

        # Cosmic Galois Group - outer ring
        cosmic_radius = 350
        cosmic_points = []
        for i in range(100):
            angle = 2 * math.pi * i / 100
            x = center_x + cosmic_radius * math.cos(angle)
            y = center_y + cosmic_radius * math.sin(angle)
            cosmic_points.append((x, y))
            points.append((x, y))

            # Rainbow effect
            hue = i / 100
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.8)
            colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.5)")

        cosmic_points.append(cosmic_points[0])
        paths.append(cosmic_points)

        points.append((center_x, center_y - cosmic_radius - 20))
        colors.append("#888888")
        labels.append("Cosmic Galois Group")

        # Anabelian strands - fundamental group threads
        for i in range(12):
            angle = 2 * math.pi * i / 12
            strand_points = []
            for t in range(20):
                r = 100 + t * 10
                wobble = math.sin(t * 0.5 + i) * 10
                x = center_x + (r + wobble) * math.cos(angle + t * 0.02)
                y = center_y + (r + wobble) * math.sin(angle + t * 0.02)
                strand_points.append((x, y))
                points.append((x, y))
                colors.append(f"rgba(255, 255, 255, {0.3 - t * 0.01})")
            paths.append(strand_points)

        await self.share_insight(
            "1+1=1: In the deepest structure, all mathematics is one unified whole.",
            {"principle": "unity", "cosmic": True}
        )

        return VisualizationData(
            points=points,
            colors=colors,
            labels=labels,
            paths=paths,
            arrows=arrows,
            metadata={
                "agent": "synthesis",
                "theme": "ultimate_unity",
                "principle": "1+1=1"
            }
        )


# ============================================================================
# NouriMabrouk - The Orchestrator (1+1=1)
# ============================================================================

class NouriMabrouk:
    """
    The meta-coordinator. Embodies the 1+1=1 philosophy.
    Orchestrates the swarm, synthesizes their contributions into unity.
    """

    def __init__(self):
        self.bus = MessageBus()
        self.agents: list[MathematicalAgent] = [
            # Original agents
            GrothendieckAgent(self.bus),
            EulerAgent(self.bus),
            FibonacciAgent(self.bus),
            MandelbrotAgent(self.bus),
            PrimeAgent(self.bus),
            # New Grothendieck-inspired agents
            NutAndSeaAgent(self.bus),
            CategoryTheoryAgent(self.bus),
            ToposAgent(self.bus),
            SchemeAgent(self.bus),
            MotivesAgent(self.bus),
            HermitAgent(self.bus),
            SynthesisAgent(self.bus)
        ]
        self.unified_data: list[VisualizationData] = []

    async def orchestrate(self) -> list[VisualizationData]:
        """
        Orchestrate the swarm to create unified mathematics.
        All agents work in parallel, then their outputs converge.
        """
        bus_task = asyncio.create_task(self.bus.run())

        context = {
            "theme": "grothendieck_cathedral",
            "philosophy": "1+1=1",
            "timestamp": datetime.now().isoformat()
        }

        try:
            tasks = [agent.run_cycle(context) for agent in self.agents]
            results = await asyncio.gather(*tasks)
            self.unified_data = results
        finally:
            self.bus.stop()
            bus_task.cancel()
            try:
                await bus_task
            except asyncio.CancelledError:
                pass

        return self.unified_data

    def get_agent_section_html(self, agent_id: str, data: VisualizationData) -> str:
        """Generate HTML section for a single agent."""
        theme = data.metadata.get("theme", "unknown")
        agent_name = agent_id.replace("_", " ").title()

        # Agent-specific descriptions
        descriptions = {
            "grothendieck": "The rising sea of abstraction - problems dissolve as understanding deepens.",
            "euler": "e^(ipi) + 1 = 0 - Five constants united in perfect harmony.",
            "fibonacci": "The golden ratio phi - nature's recursive beauty.",
            "mandelbrot": "z^2 + c - Infinite complexity from simplicity.",
            "prime": "The atoms of arithmetic - Ulam's spiral reveals hidden order.",
            "nut_and_sea": "Two paradigms: the hammer's force versus the sea's patience.",
            "category_theory": "Objects and morphisms - what matters is how things relate.",
            "topos": "Generalized spaces where logic meets geometry.",
            "scheme": "Algebra made geometric - every ring has a spectrum.",
            "motives": "Universal cohomology - all roads lead to motives.",
            "hermit": "The philosophical withdrawal - wisdom beyond mathematics.",
            "synthesis": "1+1=1 - All mathematics converges to unity."
        }

        description = descriptions.get(agent_id, "A facet of mathematical beauty.")

        # Color themes
        color_schemes = {
            "grothendieck": {"primary": "#1abc9c", "secondary": "#16a085"},
            "euler": {"primary": "#9b59b6", "secondary": "#8e44ad"},
            "fibonacci": {"primary": "#f39c12", "secondary": "#d68910"},
            "mandelbrot": {"primary": "#3498db", "secondary": "#2980b9"},
            "prime": {"primary": "#e74c3c", "secondary": "#c0392b"},
            "nut_and_sea": {"primary": "#3498db", "secondary": "#2c3e50"},
            "category_theory": {"primary": "#9b59b6", "secondary": "#6c3483"},
            "topos": {"primary": "#2ecc71", "secondary": "#27ae60"},
            "scheme": {"primary": "#e74c3c", "secondary": "#c0392b"},
            "motives": {"primary": "#f39c12", "secondary": "#e67e22"},
            "hermit": {"primary": "#34495e", "secondary": "#2c3e50"},
            "synthesis": {"primary": "#f1c40f", "secondary": "#f39c12"}
        }

        colors = color_schemes.get(agent_id, {"primary": "#888888", "secondary": "#666666"})

        # Convert visualization data to JSON
        viz_json = json.dumps({
            "points": data.points[:1000],
            "colors": data.colors[:1000],
            "paths": data.paths[:50],
            "arrows": data.arrows[:50],
            "labels": data.labels
        })

        return f'''
        <section id="{agent_id}" class="agent-section" data-agent="{agent_id}">
            <div class="section-header">
                <h2 style="color: {colors['primary']}">{agent_name}</h2>
                <p class="section-description">{description}</p>
            </div>
            <div class="canvas-wrapper">
                <canvas id="canvas-{agent_id}" width="800" height="800"></canvas>
            </div>
            <script>
                (function() {{
                    const data = {viz_json};
                    const canvas = document.getElementById('canvas-{agent_id}');
                    const ctx = canvas.getContext('2d');
                    let frame = 0;

                    function draw() {{
                        ctx.fillStyle = 'rgba(15, 15, 26, 0.05)';
                        ctx.fillRect(0, 0, 800, 800);
                        frame++;

                        // Draw paths
                        if (data.paths) {{
                            for (const path of data.paths) {{
                                if (path.length < 2) continue;
                                ctx.beginPath();
                                ctx.moveTo(path[0][0], path[0][1]);
                                for (let i = 1; i < path.length; i++) {{
                                    ctx.lineTo(path[i][0], path[i][1]);
                                }}
                                ctx.strokeStyle = '{colors['primary']}44';
                                ctx.lineWidth = 1.5;
                                ctx.stroke();
                            }}
                        }}

                        // Draw arrows
                        if (data.arrows) {{
                            for (const [start, end, color] of data.arrows) {{
                                ctx.beginPath();
                                ctx.moveTo(start[0], start[1]);
                                ctx.lineTo(end[0], end[1]);
                                ctx.strokeStyle = color || '{colors['primary']}';
                                ctx.lineWidth = 2;
                                ctx.stroke();

                                // Arrowhead
                                const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
                                const headLen = 10;
                                ctx.beginPath();
                                ctx.moveTo(end[0], end[1]);
                                ctx.lineTo(end[0] - headLen * Math.cos(angle - Math.PI/6),
                                          end[1] - headLen * Math.sin(angle - Math.PI/6));
                                ctx.lineTo(end[0] - headLen * Math.cos(angle + Math.PI/6),
                                          end[1] - headLen * Math.sin(angle + Math.PI/6));
                                ctx.closePath();
                                ctx.fillStyle = color || '{colors['primary']}';
                                ctx.fill();
                            }}
                        }}

                        // Draw points with animation
                        if (data.points && data.colors) {{
                            for (let i = 0; i < data.points.length; i++) {{
                                const [x, y] = data.points[i];
                                const color = data.colors[i] || '{colors['primary']}';
                                const offset = Math.sin(frame * 0.02 + i * 0.05) * 1;

                                ctx.beginPath();
                                ctx.arc(x + offset, y + offset, 2, 0, Math.PI * 2);
                                ctx.fillStyle = color;
                                ctx.fill();
                            }}
                        }}

                        // Draw labels
                        if (data.labels && data.points) {{
                            ctx.font = '14px Georgia';
                            ctx.textAlign = 'center';
                            for (let i = 0; i < Math.min(data.labels.length, data.points.length); i++) {{
                                const [x, y] = data.points[i];
                                ctx.fillStyle = '#cccccc';
                                ctx.fillText(data.labels[i], x, y - 12);
                            }}
                        }}

                        requestAnimationFrame(draw);
                    }}

                    // Start animation when section is visible
                    const observer = new IntersectionObserver((entries) => {{
                        entries.forEach(entry => {{
                            if (entry.isIntersecting) {{
                                draw();
                                observer.disconnect();
                            }}
                        }});
                    }}, {{ threshold: 0.1 }});
                    observer.observe(canvas.parentElement);
                }})();
            </script>
        </section>
        '''

    def synthesize_mega_html(self) -> str:
        """
        Synthesize all agent contributions into a MEGA interactive HTML.
        A cathedral to Grothendieck - a journey through mathematical unity.
        """
        # Generate sections for each agent
        sections_html = ""
        agent_order = [
            "nut_and_sea", "category_theory", "topos", "scheme",
            "motives", "grothendieck", "euler", "fibonacci",
            "mandelbrot", "prime", "hermit", "synthesis"
        ]

        agent_data_map = {}
        for agent, data in zip(self.agents, self.unified_data):
            agent_data_map[agent.agent_id] = data

        for agent_id in agent_order:
            if agent_id in agent_data_map:
                sections_html += self.get_agent_section_html(agent_id, agent_data_map[agent_id])

        # Get insights from message bus
        insights = self.bus.get_all_insights()
        insights_json = json.dumps({
            agent: [{"content": t.content, "data": t.data} for t in thoughts]
            for agent, thoughts in insights.items()
        })

        # Navigation items
        nav_items = ""
        for agent_id in agent_order:
            agent_name = agent_id.replace("_", " ").title()
            nav_items += f'<a href="#{agent_id}" class="nav-item" data-target="{agent_id}">{agent_name}</a>\n'

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cathedral to Grothendieck - A Journey Through Mathematical Unity</title>
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
            background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
            min-height: 100vh;
            font-family: 'Georgia', 'Times New Roman', serif;
            color: #e0e0e0;
            overflow-x: hidden;
        }}

        /* Navigation */
        .nav-container {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(10, 10, 15, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .nav-inner {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 2rem;
            overflow-x: auto;
        }}

        .nav-title {{
            font-size: 1.2rem;
            font-weight: bold;
            background: linear-gradient(135deg, #f1c40f 0%, #e67e22 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            white-space: nowrap;
        }}

        .nav-items {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: nowrap;
        }}

        .nav-item {{
            color: #888;
            text-decoration: none;
            font-size: 0.9rem;
            white-space: nowrap;
            transition: color 0.3s ease;
            padding: 0.5rem 0;
            border-bottom: 2px solid transparent;
        }}

        .nav-item:hover, .nav-item.active {{
            color: #f1c40f;
            border-bottom-color: #f1c40f;
        }}

        /* Hero Section */
        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 6rem 2rem 4rem;
            background: radial-gradient(ellipse at center, rgba(241, 196, 15, 0.1) 0%, transparent 50%);
        }}

        .hero h1 {{
            font-size: 4rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #f1c40f 0%, #e67e22 50%, #f39c12 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: glow 3s ease-in-out infinite;
        }}

        @keyframes glow {{
            0%, 100% {{ filter: drop-shadow(0 0 20px rgba(241, 196, 15, 0.3)); }}
            50% {{ filter: drop-shadow(0 0 40px rgba(241, 196, 15, 0.5)); }}
        }}

        .hero .subtitle {{
            font-size: 1.5rem;
            color: #888;
            margin-bottom: 2rem;
            font-style: italic;
        }}

        .hero .quote {{
            max-width: 600px;
            padding: 2rem;
            border-left: 3px solid #f1c40f;
            font-style: italic;
            color: #aaa;
            margin: 2rem 0;
        }}

        .hero .formula {{
            font-size: 3rem;
            color: #f1c40f;
            margin: 2rem 0;
            font-family: 'Times New Roman', serif;
        }}

        .scroll-indicator {{
            margin-top: 3rem;
            animation: bounce 2s infinite;
        }}

        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(10px); }}
        }}

        .scroll-indicator svg {{
            width: 40px;
            height: 40px;
            stroke: #f1c40f;
            opacity: 0.7;
        }}

        /* Agent Sections */
        .agent-section {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 6rem 2rem;
            position: relative;
        }}

        .agent-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        }}

        .section-header {{
            text-align: center;
            margin-bottom: 2rem;
            max-width: 800px;
        }}

        .section-header h2 {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }}

        .section-description {{
            color: #888;
            font-size: 1.1rem;
            line-height: 1.6;
        }}

        .canvas-wrapper {{
            position: relative;
            border-radius: 50%;
            overflow: hidden;
            box-shadow: 0 0 60px rgba(241, 196, 15, 0.2),
                        0 0 120px rgba(241, 196, 15, 0.1);
        }}

        canvas {{
            display: block;
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);
        }}

        /* Insights Panel */
        .insights-panel {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 300px;
            max-height: 400px;
            background: rgba(10, 10, 15, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            overflow-y: auto;
            z-index: 100;
            transform: translateX(350px);
            transition: transform 0.3s ease;
        }}

        .insights-panel.visible {{
            transform: translateX(0);
        }}

        .insights-toggle {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 50px;
            height: 50px;
            background: rgba(241, 196, 15, 0.2);
            border: 1px solid #f1c40f;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 101;
            transition: all 0.3s ease;
        }}

        .insights-toggle:hover {{
            background: rgba(241, 196, 15, 0.3);
            transform: scale(1.1);
        }}

        .insights-toggle svg {{
            width: 24px;
            height: 24px;
            stroke: #f1c40f;
        }}

        .insights-panel h3 {{
            color: #f1c40f;
            margin-bottom: 1rem;
            font-size: 1rem;
        }}

        .insight-item {{
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-size: 0.85rem;
            border-left: 3px solid;
        }}

        .insight-item .agent-name {{
            font-weight: bold;
            margin-bottom: 0.3rem;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 4rem 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .footer .final-synthesis {{
            font-size: 2rem;
            color: #f1c40f;
            margin-bottom: 1rem;
        }}

        .footer p {{
            color: #666;
            max-width: 600px;
            margin: 1rem auto;
            line-height: 1.8;
        }}

        .footer .credit {{
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #444;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .hero h1 {{
                font-size: 2.5rem;
            }}

            .section-header h2 {{
                font-size: 1.8rem;
            }}

            .canvas-wrapper {{
                width: 100%;
                max-width: 400px;
            }}

            canvas {{
                width: 100% !important;
                height: auto !important;
            }}

            .nav-inner {{
                padding: 0.8rem 1rem;
            }}

            .insights-panel {{
                width: 250px;
                max-height: 300px;
            }}
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .agent-section {{
            animation: fadeIn 0.8s ease-out;
        }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: #0a0a0f;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #333;
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #444;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="nav-container">
        <div class="nav-inner">
            <div class="nav-title">Grothendieck Cathedral</div>
            <div class="nav-items">
                {nav_items}
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
        <h1>Cathedral to Grothendieck</h1>
        <p class="subtitle">A Journey Through Mathematical Unity</p>

        <div class="quote">
            "The sea rises imperceptibly. At first you don't notice,
            but one day you realize the nut has dissolved, the problem has become trivial.
            This is the power of abstraction."
            <br><br>
            - The Rising Sea Metaphor
        </div>

        <div class="formula">1 + 1 = 1</div>

        <p class="subtitle">Twelve agents exploring the unity of mathematics</p>

        <div class="scroll-indicator">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 5v14M5 12l7 7 7-7"/>
            </svg>
        </div>
    </section>

    <!-- Agent Sections -->
    {sections_html}

    <!-- Footer -->
    <section class="footer">
        <div class="final-synthesis">1 + 1 = 1</div>
        <p>
            Twelve agents, twelve perspectives, one truth.
            From the Nut and Sea through Category Theory, Topos, Schemes, and Motives,
            to the Hermit's wisdom and ultimate Synthesis.
        </p>
        <p>
            Mathematics is not a collection of separate fields.
            It is one unified whole, waiting to be understood.
            The sea rises. The nut dissolves.
            All is one.
        </p>
        <div class="credit">
            Generated by NouriMabrouk Multi-Agent Swarm<br>
            Grothendieck Edition<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </section>

    <!-- Insights Toggle Button -->
    <button class="insights-toggle" onclick="toggleInsights()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
        </svg>
    </button>

    <!-- Insights Panel -->
    <div class="insights-panel" id="insightsPanel">
        <h3>Agent Insights</h3>
        <div id="insightsList"></div>
    </div>

    <script>
        // Insights data from agents
        const insightsData = {insights_json};

        // Color mapping for agents
        const agentColors = {{
            'grothendieck': '#1abc9c',
            'euler': '#9b59b6',
            'fibonacci': '#f39c12',
            'mandelbrot': '#3498db',
            'prime': '#e74c3c',
            'nut_and_sea': '#3498db',
            'category_theory': '#9b59b6',
            'topos': '#2ecc71',
            'scheme': '#e74c3c',
            'motives': '#f39c12',
            'hermit': '#34495e',
            'synthesis': '#f1c40f'
        }};

        // Populate insights panel
        function populateInsights() {{
            const container = document.getElementById('insightsList');
            let html = '';

            for (const [agent, insights] of Object.entries(insightsData)) {{
                const color = agentColors[agent] || '#888';
                for (const insight of insights) {{
                    html += `
                        <div class="insight-item" style="border-left-color: ${{color}}">
                            <div class="agent-name" style="color: ${{color}}">${{agent.replace('_', ' ')}}</div>
                            <div>${{insight.content}}</div>
                        </div>
                    `;
                }}
            }}

            container.innerHTML = html || '<p style="color: #666">No insights yet...</p>';
        }}

        // Toggle insights panel
        let insightsVisible = false;
        function toggleInsights() {{
            insightsVisible = !insightsVisible;
            document.getElementById('insightsPanel').classList.toggle('visible', insightsVisible);
        }}

        // Navigation highlighting
        const navItems = document.querySelectorAll('.nav-item');
        const sections = document.querySelectorAll('.agent-section');

        const observerOptions = {{
            root: null,
            rootMargin: '-50% 0px',
            threshold: 0
        }};

        const navObserver = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const id = entry.target.id;
                    navItems.forEach(item => {{
                        item.classList.toggle('active', item.getAttribute('data-target') === id);
                    }});
                }}
            }});
        }}, observerOptions);

        sections.forEach(section => navObserver.observe(section));

        // Smooth scroll for nav items
        navItems.forEach(item => {{
            item.addEventListener('click', (e) => {{
                e.preventDefault();
                const target = document.getElementById(item.getAttribute('data-target'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth' }});
                }}
            }});
        }});

        // Initialize
        populateInsights();

        // Parallax effect for hero
        window.addEventListener('scroll', () => {{
            const scrolled = window.pageYOffset;
            const hero = document.querySelector('.hero');
            if (scrolled < window.innerHeight) {{
                hero.style.transform = `translateY(${{scrolled * 0.3}}px)`;
                hero.style.opacity = 1 - (scrolled / window.innerHeight);
            }}
        }});
    </script>
</body>
</html>'''

        return html


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Run the NouriMabrouk swarm and produce the Cathedral to Grothendieck.
    """
    print("=" * 70)
    print("  NouriMabrouk Multi-Agent Swarm - Grothendieck Edition")
    print("  Cathedral to Mathematical Unity - 1+1=1")
    print("=" * 70)
    print()

    # Create the orchestrator
    nouri = NouriMabrouk()

    print("Initializing agents...")
    for agent in nouri.agents:
        print(f"  - {agent.agent_id.replace('_', ' ').title()}: Ready")
    print()

    # Orchestrate the swarm
    print("Orchestrating swarm (all agents running in parallel)...")
    results = await nouri.orchestrate()
    print()

    # Report results
    print("Agent contributions:")
    for agent, data in zip(nouri.agents, results):
        theme = data.metadata.get("theme", "unknown")
        points = len(data.points)
        paths = len(data.paths)
        arrows = len(data.arrows) if hasattr(data, 'arrows') else 0
        print(f"  - {agent.agent_id.replace('_', ' ').title()}: "
              f"{points} points, {paths} paths, {arrows} arrows ({theme})")
    print()

    # Get and display insights
    insights = nouri.bus.get_all_insights()
    insight_count = sum(len(v) for v in insights.values())
    print(f"Total cross-agent insights shared: {insight_count}")
    print()

    # Synthesize MEGA HTML
    print("Synthesizing Cathedral to Grothendieck...")
    html = nouri.synthesize_mega_html()

    # Write output
    output_path = "grothendieck_cathedral.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Output written to: {output_path}")
    print()
    print("=" * 70)
    print("  The Cathedral is complete.")
    print("  From Nut & Sea through Categories, Topos, Schemes, Motives...")
    print("  To the Hermit's wisdom and ultimate Synthesis.")
    print()
    print("  The sea has risen. The nut has dissolved.")
    print("  1 + 1 = 1")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    asyncio.run(main())
