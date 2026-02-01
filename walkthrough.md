# Walkthrough: Natural Head Rotation (Pseudo-3D)

## Goal
Implement a **Natural 3D-like Head Turn** (Yaw & Pitch) using vertex deformation on a standard 2D texture, replacing rigid "Parallax" sliding.

## Concepts & Evolution

### 1. Why Vertex Deformation?
-   **Old Approach (Parallax)**: Sliding flat layers (Eyes move faster than Face).
    -   *Issue*: Face looks flat; features slide off the face "plate".
-   **New Approach (Deformation)**:
    -   We treat the 2D image as a malleable surface.
    -   **Bulging**: Center points move more than edge points (Simulating curvature).
    -   **Foreshortening**: The side turning "away" is compressed (Perspective).

### 2. The Algorithm (Non-Linear Vertex Deformation)
We implemented `NonLinearParallaxDeformer` in `src/deformer.py`.

#### A. Weights (Radius-based)
We define a "Rigidity" mask where the center of the face deforms less (keeping eyes/nose structure) while the edges stretch more.
$$ Weight = 1.0 - (\frac{r}{R})^3 $$

#### B. Yaw Rotation (Left/Right)
-   **Displacement (Bulge X)**: Moves features horizontally.
    $$ dx = Yaw \times Weight \times Sensitivity $$
-   **Perspective Compression (Scale X)**:
    When turning Left (Yaw < 0), the Left side is compressed:
    $$ ScaleX = 1.0 - (Yaw \times Strength \times \frac{x}{R}) $$

#### C. Pitch Rotation (Up/Down)
-   **Displacement (Bulge Y)**: Moves features vertically.
    $$ dy = Pitch \times Weight \times Sensitivity $$
-   **Perspective Compression (Scale Y)**:
    When looking Down (Pitch > 0), the Chin (Bottom) should act "further away" relative to the forehead? Or rather, the chin tucks in.
    -   *Logic*: Looking Down -> Chin Squashes.
    $$ ScaleY = 1.0 + (Pitch \times Strength \times \frac{y}{R}) $$
    *(Note: The sign is flipped compared to Yaw due to Y-axis orientation).*

## Key Files
-   `src/deformer.py`: Contains `NonLinearParallaxDeformer` class.
-   `src/live2d.py`: Integrates deformer into `Live2DPart.update`.
-   `src/practice/face_deformer.py`: Prototype script for testing the math.

## Usage
Head rotation is automatically driven by the `tracker.py` data (MediaPipe Face Mesh).
-   **Yaw**: -30 to +30 degrees.
-   **Pitch**: -30 to +30 degrees.

No manual configuration needed beyond `deformer` initialization in `live2d.py`.
