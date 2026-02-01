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
    eye_path = os.path.join(project_root, MODEL_PATH, "MouthClose.png")

    print(f"Loading Face from: {face_path}")
    print(f"Loading Feature from: {eye_path}")

    # Create Meshes
    pts_face, tris_face = create_image_mesh(
        face_path, debug=False, max_area=500, simplify_epsilon=1.0
    )

    pts_eye, tris_eye = create_image_mesh(
        eye_path, debug=False, max_area=100, simplify_epsilon=1.0
    )

    # 2. Arrange Positions relative to Face Center (0,0)

    # Get Global Transforms from JSON
    if "Face" not in MODEL_DATA or "MouthClose" not in MODEL_DATA:
        print("Error: Missing Face or MouthClose in MODEL_DATA")
        return

    face_info = MODEL_DATA["Face"]
    eye_info = MODEL_DATA["MouthClose"]

    # Global Coordinates (Pixels in PSD/Canvas)
    # Convert to Y-Up by negating Y
    face_global = np.array([face_info["global_center_x"], face_info["global_center_y"]])
    eye_global = np.array([eye_info["global_center_x"], eye_info["global_center_y"]])

    # Vector from Face Center to Eye Center
    eye_offset_from_face = eye_global - face_global
    print(f"Feature Offset from Face: {eye_offset_from_face}")

    # Center the Face Mesh to (0,0)
    # The loaded pts_face are in local image coordinates (0,0 at top-left of crop)
    # Flip local Y as well

    # Use Image Dimensions from JSON for True Center
    # This aligns 100% with the Global Center logic
    w_face = face_info["original_width"]
    h_face = face_info["original_height"]
    # In Y-Up (flipped), center is (w/2, -h/2) because 0,0 was top-left and we negated Y
    face_local_center = np.array([w_face / 2.0, h_face / 2.0])
    pts_face -= face_local_center

    # Position the Eye Mesh
    # 1. Center it locally using ITS OWN dimensions
    w_eye = eye_info["original_width"]
    h_eye = eye_info["original_height"]
    eye_local_center = np.array([w_eye / 2.0, h_eye / 2.0])
    pts_eye -= eye_local_center

    # 2. Move to offset relative to Face
    pts_eye += eye_offset_from_face

    pts_face[:, 1] *= -1
    pts_eye[:, 1] *= -1

    # Calculate Face Radius (for deformation limits)
    # Use max of width and height to cover full face (especially for tall faces)
    face_width = w_face
    face_height = h_face

    max_dim = max(face_width, face_height)
    face_radius = max_dim * 0.75  # 0.75 of max dimension (~1.5x half-dim)

    print(f"Face Size: {face_width:.2f}x{face_height:.2f}, Radius: {face_radius:.2f}")

    # 3. Setup Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    # No invert_yaxis needed for Y-Up coordinates

    # Initialize Deformer
    # Force center to (0,0) to match logic in live2d.py and ensure rotation around the true image center
    face_deformer = NonLinearParallaxDeformer(pts_face, radius_scale=0, center=(0, 0))
    # Manually override radius with our calculated max_dim radius
    face_deformer.radius = face_radius

    def get_verts(current_pts, current_tris):
        return [current_pts[t] for t in current_tris]

    face_coll = PolyCollection(
        get_verts(pts_face, tris_face),
        edgecolors="black",
        facecolors="navajowhite",
        alpha=0.5,
        label="Face",
    )
    eye_coll = PolyCollection(
        get_verts(pts_eye, tris_eye),
        edgecolors="blue",
        facecolors="white",
        alpha=0.8,
        label="Eye",
    )

    ax.add_collection(face_coll)
    ax.add_collection(eye_coll)

    # Set bounds
    all_pts = np.vstack((pts_face, pts_eye))
    margin = 50
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 4. Animation Loop
    def update(frame):
        # Oscillate yaw between -1 and 1
        yaw = np.sin(frame * 0.1)

        # Deform Mesh using the Class Instance
        # 1. Deform Face
        new_face = face_deformer.get_deformed_vertices(yaw)

        # 2. Deform Eye (Child)
        # The eye should follow the face's deformation.
        # Instead of creating a new Deformer for the eye (which would deform it relative to its own center if initialized with eye pts),
        # we should use the FACE deformer to transform the eye points.
        # This is CRITICAL for showing correct "Binding" behavior.
        new_eye, _ = face_deformer.transform_points(pts_eye, yaw)

        # Update Plot
        face_coll.set_verts(get_verts(new_face, tris_face))
        eye_coll.set_verts(get_verts(new_eye, tris_eye))

        ax.set_title(f"Face Deformation (Yaw: {yaw:.2f})")
        return face_coll, eye_coll

    ani = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
    output_path = os.path.join(src_dir, "practice", "face_turn.gif")
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, writer="pillow", fps=20)
    print("Done.")


if __name__ == "__main__":
    main()
