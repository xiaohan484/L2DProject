# Plan: Natural Head Turning Effect (Replacing Parallax)

## 1. Current Status
- **Adopted Solution**: **Non-Linear Vertex Deformation**.
    - Replaces broad geometric projections (Cylinder/Plane) with vertex-specific displacement.
    - **Effect**: "Bulging" center + "Compressing" sides.

## 2. **The Logic (Implemented in Prototype)**
    -   **Centering**: Align mesh to (0,0).
    -   **Radial Weighting**:
        -   $Weight = 1.0 - (\frac{r}{R})^3$ (Plateau Curve).
        -   Flattens the center to keep facial features rigid, pushing distortion to edges.
    -   **Displacement (Bulge)**:
        -   $dx = Yaw \cdot Weight \cdot Sensitivity$
        -   Sensitivity tuned to 60.0 (from 80.0) to reduce protrusion.
    -   **Asymmetric Compression (Perspective Trick)**:
    -   Simulates the far side turning away.
    -   $Scale = 1.0 - (Yaw \cdot Strength \cdot \frac{x}{R})$
    -   Result: Turning Left (Yaw < 0) -> Left side (Far) compresses, Right side (Near) expands.

## 3. Implementation Steps
- [x] **Prototype**: `practice/face_deformer.py` updated and verified.
- [x] **Algorithm Code**:
    -   `calculate_weights()` based on radius.
    -   `apply_deformation()` with displacement + compression.
- [x] Integration into `live2d.py`.
    - [x] Create `NonLinearParallaxDeformer` class in `live2d.py` or new file.
        - [x] Centered coordinate system.
        - [x] Weight Function: $1 - r^3$.
        - [x] Displacement + Compression logic.
    - [x] Calculate `radius` in `Face` initialization.
    - [x] Apply deformation to `Face` vertices in `update()`.
    - [x] Apply deformation to `Child Part` positions (Offset calculation).
