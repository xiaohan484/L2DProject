# Task: Implement Pitch Rotation (Up/Down)

## Objective
Extend the `NonLinearParallaxDeformer` to support vertical head rotation (Pitch), using logic symmetrical to the existing Yaw implementation.

## Steps
- [x] **Core Logic (`deformer.py`)**
    - [x] Update `get_deformed_vertices` and `get_deformed_point` signatures to accept `pitch`.
    - [x] Implement Vertical Displacement (Bulge Y).
    - [x] Implement Asymmetric Vertical Compression (Foreshortening Y).
- [x] **Integration (`live2d.py`)**
    - [x] Retrieve `Pitch` from tracker data in `Live2DPart.update`.
    - [x] Normalize Pitch (similar to Yaw).
    - [x] Pass `normalized_pitch` to `deformer` methods.
- [ ] **Verification**
    - [ ] Run `mainArcade.py`.
    - [ ] Check if looking up/down creates natural deformation.
