import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
import sys
import os
from pathlib import Path

# Setup paths to import from src
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from create_mesh import create_image_mesh
from deformer import NonLinearParallaxDeformer
from Const import MODEL_PATH, MODEL_DATA


def main():
    # 1. Load Face Mesh
    project_root = src_dir.parent
    face_path = os.path.join(project_root, MODEL_PATH, "Face.png")

    print(f"Loading Face from: {face_path}")

    # Create Meshes
    pts_face, tris_face = create_image_mesh(
        face_path, debug=False, max_area=500, simplify_epsilon=1.0
    )

    # 2. Arrange Positions relative to Face Center (0,0)
    if "Face" not in MODEL_DATA:
        print("Error: Missing Face in MODEL_DATA")
        return

    face_info = MODEL_DATA["Face"]

    # Use Image Dimensions from JSON for True Center
    w_face = face_info["original_width"]
    h_face = face_info["original_height"]
    face_local_center = np.array([w_face / 2.0, h_face / 2.0])

    # Center mesh
    pts_face -= face_local_center
    # Flip Y
    pts_face[:, 1] *= -1

    # Calculate Radius
    max_dim = max(w_face, h_face)
    face_radius = max_dim * 0.75

    print(f"Face Size: {w_face:.2f}x{h_face:.2f}, Radius: {face_radius:.2f}")

    # 3. Setup Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # Initialize Deformer
    face_deformer = NonLinearParallaxDeformer(pts_face, radius_scale=0, center=(0, 0))
    face_deformer.radius = face_radius

    # Add Grid Lines (Horizontal and Vertical) for better visualization of distortion
    # Bounds
    min_x, max_x = pts_face[:, 0].min(), pts_face[:, 0].max()
    min_y, max_y = pts_face[:, 1].min(), pts_face[:, 1].max()

    grid_lines_h = []
    for y in np.linspace(min_y, max_y, 10):
        # Line from min_x to max_x at y
        grid_lines_h.append(np.array([np.linspace(min_x, max_x, 20), np.full(20, y)]).T)

    grid_lines_v = []
    for x in np.linspace(min_x, max_x, 10):
        # Line from min_y to max_y at x
        grid_lines_v.append(np.array([np.full(20, x), np.linspace(min_y, max_y, 20)]).T)

    def get_verts(current_pts, current_tris):
        return [current_pts[t] for t in current_tris]

    face_coll = PolyCollection(
        get_verts(pts_face, tris_face),
        edgecolors="black",
        facecolors="navajowhite",
        alpha=0.5,
        label="Face",
    )

    # Lines collections
    line_colls = []
    for line in grid_lines_h + grid_lines_v:
        (lc,) = ax.plot([], [], "r-", alpha=0.5)
        line_colls.append((lc, line))

    ax.add_collection(face_coll)

    # Set bounds
    margin = 50
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # Draw initial circle bounds
    circle = plt.Circle((0, 0), face_radius, color="g", fill=False, linestyle="--")
    ax.add_artist(circle)

    ax.set_title("Pitch Deformation Test")

    # 4. Animation Loop
    def update(frame):
        # Oscillate pitch between -1.5 and 1.5
        pitch = np.sin(frame * 0.1) * 1.5

        # Deform Mesh
        new_face = face_deformer.get_deformed_vertices(yaw=0.0, pitch=pitch)

        # Update Mesh
        face_coll.set_verts(get_verts(new_face, tris_face))

        # Update Grid Lines
        for lc, line_pts in line_colls:
            deformed_line, _ = face_deformer.transform_points(
                line_pts, yaw=0.0, pitch=pitch
            )  # returns (pts, scales)
            lc.set_data(deformed_line[:, 0], deformed_line[:, 1])

        ax.set_title(f"Pitch Deformation (Pitch: {pitch:.2f})")
        return [face_coll] + [lc for lc, _ in line_colls]

    ani = FuncAnimation(fig, update, frames=63, interval=50, blit=True)
    output_path = os.path.join(src_dir, "practice", "pitch_test.gif")
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, writer="pillow", fps=20)
    print("Done.")


if __name__ == "__main__":
    main()
