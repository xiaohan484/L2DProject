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

### 3. Architecture Refactor: Deformer Stack
To support multiple types of deformation (e.g., Expression + Perspective), we refactored `Live2DPart` to use a **Deformer Stack**.
- **Before**: `Live2DPart` had a single `self.deformer` (hardcoded for Perspective).
- **After**: `Live2DPart` has `self.deformers = []`. Vertices are passed through each deformer sequentially.
- **Child Attachment**: Child parts attach to the parent's *deformed surface* by running their anchor point through the parent's deformer stack.

## Key Files
- `src/deformer_system.py`: (Was `deformers.py`) System interface and wrappers.
- `src/parallax_deformer.py`: (Was `deformer.py`) Core math for parallax.
- `src/live2d.py`: Integrates `Deformer Stack` into `Live2DPart.update`.
- `tests/test_deformer.py`: Unit tests for deformation math.
- `tests/test_live2d_part.py`: Unit tests for hierarchy and stack logic.

## Usage
Head rotation is automatically driven by the `tracker.py` data (MediaPipe Face Mesh).
-   **Yaw**: -30 to +30 degrees.
-   **Pitch**: -30 to +30 degrees.

No manual configuration needed beyond `deformer` initialization in `live2d.py`.
