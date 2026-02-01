import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import triangle as tr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
import time

# Get the absolute path of the directory one level up
# __file__ is the path to the current script
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
from Const import MODEL_PATH


def create_image_mesh(
    image_path, debug=True, max_area=100, min_angle=30, simplify_epsilon=0
):
    # 1. Load image and convert to grayscale
    # We use OpenCV to detect edges
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = img[:, :, 3] == 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img[mask] = [255, 255, 255, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray[mask] = 255
    ret, binary_img = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4. Draw results (Optional)
    max_contour = max(contours, key=cv2.contourArea)

    # Optional: Simplify contour to reduce vertex count (Smoothing)
    if simplify_epsilon > 0:
        max_contour = cv2.approxPolyDP(max_contour, simplify_epsilon, True)

    # Triangulate
    # 'p': Planar Straight Line Graph (Constrained Delaunay)
    # 'q30': Quality mesh (no angle less than 30 degrees)
    # 'a500': Max area of triangles (adjust this to change density)
    # 'D': Conforming Delaunay
    vertices = max_contour.reshape(-1, 2).astype(np.float64)
    n = len(vertices)
    segments = np.array([[i, (i + 1) % n] for i in range(n)])
    input_data = {"vertices": vertices, "segments": segments}

    # Construct options string
    # e.g. "pq30a100"
    opts = f"pa{max_area}"
    if min_angle > 0:
        opts = f"pq{min_angle}a{max_area}"

    mesh = tr.triangulate(input_data, opts)
    pts = mesh["vertices"]
    tris = mesh["triangles"]
    # Plot the mesh
    if debug is True:
        plt.imshow(img)
        plt.triplot(pts[:, 0], pts[:, 1], tris, color=(0, 1, 0, 0.5), linewidth=0.5)
        plt.title("Constrained Delaunay Mesh on Hair")
        plt.axis("off")
        plt.show()
    return pts, tris


# Assuming pts is (N, 3) and tris is (M, 3)
def get_edges(tris):
    # Extract unique edges from triangles
    edges = set()
    for t in tris:
        for i in range(3):
            v1, v2 = sorted([t[i], t[(i + 1) % 3]])
            edges.add((v1, v2))
    return np.array(list(edges))


class PBDCloth2D:
    def __init__(
        self,
        pts,
        tris,
        stiffness=0.1,
        dt=0.016,
        fixed_ratio=0.1,
        damping=0.99,
        lra_stiffness=0.8,
    ):
        # pts shape: (N, 2)
        self.pos = pts.astype(np.float64)
        self.vel = np.zeros_like(self.pos)
        self.inv_mass = np.ones(len(self.pos))
        self.stiffness = stiffness  # Adjustable stiffness

        # Fixed the top points (top fixed_ratio)
        max_y = np.max(self.pos[:, 1])
        min_y = np.min(self.pos[:, 1])
        height = max_y - min_y
        self.fixed_indices = np.where(self.pos[:, 1] >= max_y - height * fixed_ratio)[0]
        self.inv_mass[self.fixed_indices] = 0.0

        self.tris = tris
        self.edges = self._get_edges(tris)
        self.rest_lengths = np.linalg.norm(
            self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]], axis=1
        )

        self.dt = dt
        self.gravity = np.array([0.0, -9.8])
        self.damping = damping
        self.lra_stiffness = lra_stiffness

        # Bending Constraints
        self.bending_stiffness = stiffness * 0.5  # Usually softer than structural
        self.bending_edges = self._get_bending_edges(tris)
        if len(self.bending_edges) > 0:
            self.bending_rest_lengths = np.linalg.norm(
                self.pos[self.bending_edges[:, 0]] - self.pos[self.bending_edges[:, 1]],
                axis=1,
            )
        else:
            self.bending_rest_lengths = np.array([])

        # LRA (Long Range Attachments) Initialization
        # For each particle, find the closest fixed particle (anchor)
        # Store (particle_idx, anchor_idx, rest_length)
        self.lra_indices = []  # List of [particle_idx, anchor_idx]
        self.lra_rest_lengths = []

        # Only build LRA if we have fixed points
        if len(self.fixed_indices) > 0:
            fixed_pos = self.pos[self.fixed_indices]
            for i in range(len(self.pos)):
                # Skip if it is a fixed point itself
                if self.inv_mass[i] == 0:
                    continue

                # Find closest fixed point
                dists = np.linalg.norm(fixed_pos - self.pos[i], axis=1)
                best_fixed_idx = self.fixed_indices[np.argmin(dists)]
                dist = np.min(dists)

                self.lra_indices.append([i, best_fixed_idx])
                self.lra_rest_lengths.append(dist)

            self.lra_indices = np.array(self.lra_indices)
            self.lra_rest_lengths = np.array(self.lra_rest_lengths)
        else:
            self.lra_indices = np.array([])
            self.lra_rest_lengths = np.array([])

    def _get_edges(self, tris):
        edges = set()
        for t in tris:
            for i in range(3):
                v1, v2 = sorted([t[i], t[(i + 1) % 3]])
                edges.add((v1, v2))
        return np.array(list(edges))

    def _get_bending_edges(self, tris):
        # Find adjacent triangles and connect their opposite vertices
        # 1. Build an edge-to-triangle map
        edge_to_tris = {}
        for t_idx, t in enumerate(tris):
            for i in range(3):
                v1, v2 = sorted([t[i], t[(i + 1) % 3]])
                edge = (v1, v2)
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(
                    (t_idx, t[(i + 2) % 3])
                )  # Store tri_idx and opposite vertex

        bending_edges = set()
        for edge, info in edge_to_tris.items():
            if len(info) == 2:
                # Two triangles share this edge -> Interior edge
                # Connect the two opposite vertices
                v_op1 = info[0][1]
                v_op2 = info[1][1]
                if v_op1 != v_op2:
                    bending_edges.add(tuple(sorted((v_op1, v_op2))))

        if not bending_edges:
            return np.array([])
        return np.array(list(bending_edges))

    def _solve_distance_constraints_vectorized(self, p):
        # 1. Fetch all edge endpoint coordinates at once
        # self.edges shape: (M, 2)
        p1 = p[self.edges[:, 0]]
        p2 = p[self.edges[:, 1]]

        # 2. 計算所有邊的當前長度
        deltas = p1 - p2  # Shape: (M, 2)
        dists = np.linalg.norm(deltas, axis=1)  # Shape: (M,)

        # 3. 計算誤差
        # Avoid division by zero
        dists = np.where(dists < 1e-9, 1e-9, dists)

        # 4. 計算修正量
        # w_sum = w1 + w2
        w1 = self.inv_mass[self.edges[:, 0]]
        w2 = self.inv_mass[self.edges[:, 1]]
        w_sum = w1 + w2

        # 只處理 w_sum > 0 的邊
        mask = w_sum > 0
        corrections = np.zeros_like(deltas)
        # Correction formula: delta_p = (w / w_sum) * diff * deltas
        # diff = (dists - rest_lengths) / dists * stiffness

        diff = (dists[mask] - self.rest_lengths[mask]) / dists[mask] * self.stiffness
        corrections[mask] = (diff / w_sum[mask])[:, np.newaxis] * deltas[mask]

        # 5. Update positions (tricky part: a single point may belong to multiple edges)
        # Using np.add.at safely accumulates multiple corrections to the same point
        # print(corrections)
        np.add.at(p, edges[:, 0], -w1[:, np.newaxis] * corrections)
        np.add.at(p, edges[:, 1], w2[:, np.newaxis] * corrections)

    def _solve_constraints_generic(self, p, edges, rest_lengths, stiffness):
        if len(edges) == 0:
            return

        p1 = p[edges[:, 0]]
        p2 = p[edges[:, 1]]

        deltas = p1 - p2
        dists = np.linalg.norm(deltas, axis=1)
        dists = np.where(dists < 1e-9, 1e-9, dists)

        w1 = self.inv_mass[edges[:, 0]]
        w2 = self.inv_mass[edges[:, 1]]
        w_sum = w1 + w2

        mask = w_sum > 0

        # Calculate corrections
        diff = (dists[mask] - rest_lengths[mask]) / dists[mask] * stiffness
        corrections = (diff / w_sum[mask])[:, np.newaxis] * deltas[mask]

        np.add.at(p, edges[mask, 0], -w1[mask][:, np.newaxis] * corrections)
        np.add.at(p, edges[mask, 1], w2[mask][:, np.newaxis] * corrections)

    def _solve_distance_constraints(self, p):
        for i, (idx1, idx2) in enumerate(self.edges):
            w1, w2 = self.inv_mass[idx1], self.inv_mass[idx2]
            if w1 + w2 == 0:
                continue

            delta = p[idx1] - p[idx2]
            dist = np.linalg.norm(delta)
            if dist < 1e-9:
                continue

            correction = (dist - self.rest_lengths[i]) / (w1 + w2) * (delta / dist)

            # Since inv_mass[fixed] is 0, the correction automatically becomes 0
            if w1 > 0:
                p[idx1] -= w1 * correction
            if w2 > 0:
                p[idx2] += w2 * correction

    def _solve_lra_constraints(self, p):
        if len(self.lra_indices) == 0:
            return

        # p_i: moving particles, p_a: anchor particles
        p_i = p[self.lra_indices[:, 0]]
        p_a = p[self.lra_indices[:, 1]]

        deltas = p_i - p_a
        dists = np.linalg.norm(deltas, axis=1)
        dists = np.where(dists < 1e-9, 1e-9, dists)

        # LRA is unilateral: only correct if stretched (current dist > rest dist)
        # But for hair keeping shape, bilateral (keeping exact distance) might help against curling too?
        # Standard LRA is usually unilateral (dist > rest).
        # Let's try bilateral first (dist != rest) to keep structure stronger?
        # No, bilateral makes it rigid like a stick. Unilateral allows bending but prevents stretching.
        # But user wants to prevent "curling" (shortening).
        # Actually, if hair curls up, its distance to root decreases?
        # Yes! So if we enforce dist >= rest_length, we prevent curling up?
        # Usually LRA enforces dist <= rest_length (prevent stretch).
        # To prevent curling up, we would need dist >= rest_length... but that prevents bending up.
        # Let's start with standard LRA (prevent stretch) as it stabilizes the system generally.
        # It indirectly helps curling by adding a strong skeleton.

        # We will use Unilateral: Only correct if dist > rest_length
        mask = dists > self.lra_rest_lengths

        w = self.inv_mass[self.lra_indices[:, 0]]
        # Anchors have inv_mass 0, so w_sum = w_particle

        # Correction
        # diff = (dists - rest) / dists
        # delta_p = -w * diff * deltas (move particle towards anchor)

        diff = (dists[mask] - self.lra_rest_lengths[mask]) / dists[mask]
        # Stiffness? LRA is usually stiff. Let's use 1.0 or user configurable.
        # We can use self.stiffness or a separate lra_stiffness.
        corrections = (diff * self.lra_stiffness * w[mask])[:, np.newaxis] * deltas[
            mask
        ]

        # Apply correction to the moving particle
        np.add.at(p, self.lra_indices[mask, 0], -corrections)

    def step(self, iterations=8, wind_strength=0.0, time_val=0.0):
        # Create a mask to identify points that are "movable"
        moving_mask = self.inv_mass > 0

        # --- 1. Prediction stage: Apply gravity and velocity only to movable points ---
        # Initialize predicted position with current position
        p = self.pos.copy()

        # Apply external forces (Gravity + Wind)
        # Wind: Sinusoidal wave traveling down the hair (y-axis)
        # Force is primarily in X direction
        if wind_strength != 0:
            # Frequency and Wave number can be tuned or passed as args
            # sin(time * speed + y * scale)
            wind_force = np.sin(time_val * 3.0 + self.pos[:, 1] * 0.05) * wind_strength
            # Add random turbulence?
            # wind_force += (np.random.rand(len(self.pos)) - 0.5) * wind_strength * 0.2

            self.vel[moving_mask, 0] += wind_force[moving_mask] * self.dt

        # Update non-fixed points only
        self.vel[moving_mask] += self.gravity * self.dt
        self.vel[moving_mask] *= self.damping
        p[moving_mask] += self.vel[moving_mask] * self.dt
        for _ in range(iterations):
            # Structural constraints
            self._solve_constraints_generic(
                p, self.edges, self.rest_lengths, self.stiffness
            )
            # Bending constraints
            self._solve_constraints_generic(
                p, self.bending_edges, self.bending_rest_lengths, self.bending_stiffness
            )
            # LRA Constraints
            self._solve_lra_constraints(p)

        self.vel[moving_mask] = (p[moving_mask] - self.pos[moving_mask]) / self.dt
        self.vel[~moving_mask] = 0.0  # Force velocity of fixed points to 0

        self.pos = p.copy()


if __name__ == "__main__":
    # --- 1. Initialize data ---
    # Create a 2D vertical grid
    pts, tris = create_image_mesh(
        f"{MODEL_PATH}/FaceLandmark.png", False, max_area=500, simplify_epsilon=1.0
    )
    pts[:, 1] = -pts[:, 1]
    sim = PBDCloth2D(pts, tris)

    # --- 2. 設定動畫畫布 ---
    fig, ax = plt.subplots(figsize=(6, 8))
    # Important: Set 1:1 aspect ratio
    margin = 1.0
    ax.set_aspect("equal")

    # Use PolyCollection for efficient updates
    # vertices shape needs to be (N_tris, 3, 2)
    def get_verts():
        return [sim.pos[t] for t in sim.tris]

    poly_coll = PolyCollection(
        get_verts(), edgecolors="blue", facecolors="skyblue", alpha=0.5
    )
    ax.add_collection(poly_coll)

    def update(frame):
        # 1. Find indices of fixed points (where inv_mass == 0)
        fixed_mask = sim.inv_mass == 0

        # 2. Move fixed points horizontally over time (e.g., using a sine wave)
        # Move the anchor points using a sine wave
        time_val = frame * 0.1
        move_x = np.sin(time_val) * 0.5 * 100  # Move range: -0.5 to 0.5

        # Note: Modifying sim.pos directly treats it as "ground truth" that the solver adapts to
        # Update the x-coordinate of fixed points
        sim.pos[fixed_mask, 0] += move_x * 0.1  # Add a bit of displacement here
        x_min, x_max = sim.pos[:, 0].min() - margin, sim.pos[:, 0].max() + margin
        y_min, y_max = (
            sim.pos[:, 1].min() - 5.0,
            sim.pos[:, 1].max() + margin,
        )  # Leave extra space at the bottom for the hair to hang down

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 3. Perform physics simulation
        # The solver will now see that constraints are violated and pull the rest

        start_sim = time.perf_counter()
        for _ in range(2):
            sim.step()
        end_sim = time.perf_counter()
        print(sim.pos[fixed_mask])

        # 4. Update plot data

        start_render = time.perf_counter()
        poly_coll.set_verts(get_verts())
        end_render = time.perf_counter()
        print(f"Sim: {end_sim-start_sim:.4f}s, Render: {end_render-start_render:.4f}s")
        return (poly_coll,)

    # --- 3. Start animation ---
    ani = FuncAnimation(fig, update, frames=200, interval=20, blit=True)
    plt.title("2D PBD Cloth Simulation (1:1 Aspect)")
    plt.show()
