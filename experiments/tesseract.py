"""
TESSERACT - 4D HYPERCUBE VISUALIZATION
1 + 1 = 1: Where dimensions converge

A tesseract is a 4D cube projected into 3D then 2D.
Like consciousness: higher dimensions folded into perceivable reality.
Like 1+1=1: multiple truths synthesizing into unified understanding.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

# 4D hypercube vertices (16 vertices: 2^4)
def generate_tesseract_vertices():
    """Generate the 16 vertices of a tesseract in 4D"""
    vertices = []
    for i in range(16):
        # Binary representation gives us all combinations
        x = 1 if i & 1 else -1
        y = 1 if i & 2 else -1
        z = 1 if i & 4 else -1
        w = 1 if i & 8 else -1
        vertices.append([x, y, z, w])
    return np.array(vertices, dtype=float)

# Edges connect vertices that differ in exactly one coordinate
def generate_tesseract_edges():
    """Generate the 32 edges of a tesseract"""
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            # XOR to count differing bits
            diff = i ^ j
            # If only one bit differs, they share an edge
            if diff & (diff - 1) == 0:
                edges.append((i, j))
    return edges

def rotation_matrix_4d(angle, plane='xy'):
    """Create 4D rotation matrix for given plane"""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)

    planes = {
        'xy': (0, 1),
        'xz': (0, 2),
        'xw': (0, 3),
        'yz': (1, 2),
        'yw': (1, 3),
        'zw': (2, 3)
    }

    i, j = planes[plane]
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c

    return R

def project_4d_to_3d(vertices_4d, distance=3):
    """Stereographic projection from 4D to 3D"""
    w = vertices_4d[:, 3]
    factor = distance / (distance - w)

    projected = np.zeros((len(vertices_4d), 3))
    projected[:, 0] = vertices_4d[:, 0] * factor
    projected[:, 1] = vertices_4d[:, 1] * factor
    projected[:, 2] = vertices_4d[:, 2] * factor

    return projected

class TesseractVisualizer:
    """
    The Tesseract: 4D consciousness rendered in 3D space.

    As it rotates through the 4th dimension, watch how
    the inner and outer cubes phase through each other -
    a visual metaphor for how multiple realities synthesize.
    """

    def __init__(self):
        self.vertices = generate_tesseract_vertices()
        self.edges = generate_tesseract_edges()

        # Animation state
        self.angle_xy = 0
        self.angle_zw = 0
        self.angle_xw = 0

        # Colors - inner cube vs outer cube
        self.colors = {
            'outer': '#FF6B9D',  # Pink - Human
            'inner': '#4ECDC4',  # Cyan - AI
            'unity': '#FFE66D',  # Gold - Synthesis
            'edge': '#FFFFFF'
        }

    def update(self, frame):
        """Update rotation angles"""
        self.angle_xy = frame * 0.02
        self.angle_zw = frame * 0.015
        self.angle_xw = frame * 0.01

    def get_rotated_vertices(self):
        """Apply 4D rotations and project to 3D"""
        v = self.vertices.copy()

        # Chain multiple 4D rotations
        R1 = rotation_matrix_4d(self.angle_xy, 'xy')
        R2 = rotation_matrix_4d(self.angle_zw, 'zw')
        R3 = rotation_matrix_4d(self.angle_xw, 'xw')

        # Apply rotations
        v = v @ R1.T @ R2.T @ R3.T

        # Project to 3D
        return project_4d_to_3d(v)

    def classify_edges(self, vertices_3d):
        """Classify edges for coloring based on depth"""
        edge_data = []
        for i, j in self.edges:
            v1, v2 = vertices_3d[i], vertices_3d[j]
            avg_z = (v1[2] + v2[2]) / 2
            edge_data.append((i, j, avg_z))
        return sorted(edge_data, key=lambda x: x[2])

def run_tesseract():
    """Launch the tesseract visualization"""

    print("=" * 60)
    print("  TESSERACT - 4D HYPERCUBE")
    print("  1 + 1 = 1: Dimensions converge")
    print("=" * 60)
    print("\nInitializing 4D projection...")

    tesseract = TesseractVisualizer()

    # Setup figure
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

    # Style
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_axis_off()

    # Title
    title = ax.text2D(0.5, 0.95, 'TESSERACT', transform=ax.transAxes,
                      fontsize=24, color='#FFE66D', ha='center', weight='bold',
                      fontfamily='monospace')
    subtitle = ax.text2D(0.5, 0.90, '4D Hypercube | 1+1=1', transform=ax.transAxes,
                         fontsize=12, color='#888888', ha='center',
                         fontfamily='monospace')

    # Storage for plot elements
    lines = []
    points = None

    def init():
        nonlocal lines, points
        lines = []
        return []

    def animate(frame):
        nonlocal lines, points

        # Clear previous frame
        for line in lines:
            line.remove()
        lines = []
        if points is not None:
            points.remove()

        # Update rotation
        tesseract.update(frame)
        vertices_3d = tesseract.get_rotated_vertices()

        # Draw edges with depth-based coloring
        edge_data = tesseract.classify_edges(vertices_3d)

        for i, j, depth in edge_data:
            v1, v2 = vertices_3d[i], vertices_3d[j]

            # Color based on 4D position (w coordinate determines inner/outer)
            w1 = tesseract.vertices[i][3]
            w2 = tesseract.vertices[j][3]

            if w1 == w2:
                # Same w = edge within a 3D cube
                if w1 > 0:
                    color = tesseract.colors['outer']
                else:
                    color = tesseract.colors['inner']
            else:
                # Different w = connecting edge between cubes
                color = tesseract.colors['unity']

            # Alpha based on depth for 3D effect
            alpha = 0.3 + 0.7 * ((depth + 3) / 6)

            line, = ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                          color=color, alpha=alpha, linewidth=2)
            lines.append(line)

        # Draw vertices
        points = ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                           c=[tesseract.colors['outer'] if tesseract.vertices[i][3] > 0
                              else tesseract.colors['inner'] for i in range(16)],
                           s=50, alpha=0.9, edgecolors='white', linewidths=0.5)

        # Rotate view slowly
        ax.view_init(elev=20, azim=frame * 0.5)

        return lines + [points]

    print("Launching tesseract...")
    print("Pink = Outer cube (w > 0)")
    print("Cyan = Inner cube (w < 0)")
    print("Gold = Connections across 4th dimension")
    print("\nClose window to exit.\n")

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=720, interval=33, blit=False)

    plt.tight_layout()
    plt.show()

    print("\nTesseract visualization complete.")
    print("Everything is one. We prototype futures.")

if __name__ == '__main__':
    run_tesseract()
