import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
import sys
import os
from pathlib import Path

# Setup paths to import from src
# Assuming this script is in e:\Live2DProject\src\practice\face_deformer.py
# create_mesh is in e:\Live2DProject\src\create_mesh.py
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from create_mesh import create_image_mesh
from Const import MODEL_PATH


def deform_mesh(pts, yaw, radius=None):
    """
    Non-Linear Vertex Deformation for Head Rotation.
    """
    # 1. Centering
    min_vals = pts.min(axis=0)
    max_vals = pts.max(axis=0)
    center = (min_vals + max_vals) / 2.0
    centered_pts = pts - center

    xs = centered_pts[:, 0]
    ys = centered_pts[:, 1]

    # 2. Radius & Weights
    if radius is None:
        width = max_vals[0] - min_vals[0]
        # Use a slightly loose radius to ensure edges don't clamp too abruptly
        calc_radius = (width / 2.0) * 1.2
    else:
        calc_radius = radius

    if calc_radius == 0:
        return pts

    dists = np.sqrt(xs**2 + ys**2)
    norm_dists = dists / calc_radius
    norm_dists = np.clip(norm_dists, 0.0, 1.0)

    # Weight Function:
    # Previous: Cosine (Round peak).
    # New: Flattened Peak (Plateau).
    # This keeps the central face (eyes/nose/mouth) moving more rigidly together,
    # pushing the distortion to the edges.
    # Exponent higher = Flatter center, sharper drop at edge.
    vis_param_power = 3.0
    weights = 1.0 - (norm_dists**vis_param_power)

    # 3. Displacement (Parallax Bulge)
    # Reduced sensitivity to avoid over-protrusion
    sensitivity = 60.0  # Pixels
    dx = yaw * weights * sensitivity

    # 4. Asymmetric Compression (Perspective Trick)
    # Simulate foreshortening on the "far" side.
    # Yaw < 0 (Look Left) -> Left side (x < 0) is Far -> Compress.
    # Formula: s = 1.0 - Yaw * Strength * (x / R)
    # Check: Yaw=-1, x=-R -> s = 1 - (-1)*K*(-1) = 1 - K (Compress).

    compression_strength = 0.4  # Tune this (0.0 to 1.0)

    # Normalize X for compression gradient (-1 to 1)
    # Use current xs or displaced xs?
    # Usually better to use original positions for stable gradient.
    x_norm_signed = xs / calc_radius
    x_norm_signed = np.clip(x_norm_signed, -1.0, 1.0)

    scale_factor = 1.0 + (yaw * compression_strength * x_norm_signed)
    # Note: "+" because we want Yaw(-1)*x(-1) to be Positive to INCREASE scale?
    # User said: "Left side should scale closer together (Compress)".
    # Compress means distance between points gets smaller.
    # If I multiply position by 0.8, the whole grid shrinks towards center.
    # If I multiply Left side by 0.8 and Right side by 1.2...

    # Let's re-verify the prompt requirement for sign.
    # "Turning Left (negative Yaw), vertices on the left side should scale closer together"
    # Left side (x < 0). Yaw < 0.
    # If I use `1 + Yaw * S * x_norm`:
    # Yaw=-1, x=-1 -> 1 + (-1)*S*(-1) = 1 + S. (Expand). WRONG.
    # So it must be MINUS.
    # `1 - Yaw * S * x_norm`
    # Yaw=-1, x=-1 -> 1 - (-1)*S*(-1) = 1 - S. (Compress). CORRECT.

    scale_factor = 1.0 - (yaw * compression_strength * x_norm_signed)

    # Apply Displacement THEN Compression?
    # Or Compression THEN Displacement?
    # Usually Perspective (Compression) applies to the final view.

    temp_xs = xs + dx
    new_xs = temp_xs * scale_factor
    new_ys = ys  # Y usually unchanged or slight scale?

    # Restore
    new_pts = np.column_stack((new_xs, new_ys)) + center
    return new_pts


def main():
    # 1. Load Face Mesh
    # Back up to project root for assets if running from src/practice
    project_root = src_dir.parent
    image_path = os.path.join(project_root, MODEL_PATH, "Face.png")

    print(f"Loading image from: {image_path}")

    # Create Mesh
    # Use larger max_area to get fewer triangles for clearer debugging initially
    pts, tris = create_image_mesh(
        image_path, debug=False, max_area=500, simplify_epsilon=1.0
    )

    # Center the mesh for easier math
    center_mean = np.mean(pts, axis=0)
    pts -= center_mean

    # Calculate Radius from Face Mesh
    xs_face = pts[:, 0]
    width = xs_face.max() - xs_face.min()
    face_radius = width * 0.75
    print(f"Face Width: {width:.2f}, Radius: {face_radius:.2f}")

    # Define a "Control Point" / "Anchor" for an eye (e.g., Left Eye position approx)
    # Assume eye is somewhat to the left and up
    eye_pos = np.array([[-50.0, 50.0]])  # Shape (1, 2)

    # 2. Setup Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # Initial Draw
    def get_verts(current_pts):
        return [current_pts[t] for t in tris]

    poly_coll = PolyCollection(
        get_verts(pts), edgecolors="black", facecolors="skyblue", alpha=0.5
    )
    ax.add_collection(poly_coll)

    # Landmark/Anchor dot
    (dot,) = ax.plot([], [], "ro", markersize=10, label="Left Eye Anchor")
    ax.legend()

    # Set bounds
    margin = 50
    x_min, x_max = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    y_min, y_max = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 3. Animation Loop
    def update(frame):
        # Oscillate yaw between -1 and 1
        yaw = np.sin(frame * 0.1)

        # Deform Mesh
        new_pts = deform_mesh(pts, yaw, face_radius)

        # Deform Anchor (Apply same logic)
        new_eye = deform_mesh(eye_pos, yaw, face_radius)

        # Update Plot
        poly_coll.set_verts(get_verts(new_pts))
        dot.set_data(new_eye[:, 0], new_eye[:, 1])

        ax.set_title(f"Face Deformation (Yaw: {yaw:.2f})")
        return poly_coll, dot

    # ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    # plt.show()

    ani = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
    output_path = os.path.join(src_dir, "practice", "face_turn.gif")
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, writer="pillow", fps=20)
    print("Done.")


if __name__ == "__main__":
    main()
