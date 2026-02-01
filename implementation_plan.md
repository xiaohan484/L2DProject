# Implementation Plan - Pitch Rotation

## Goal
Implement vertical head rotation (Pitch) using the Non-Linear Vertex Deformation method, enabling the face to "look up" and "look down" with natural volume.

## Proposed Changes

### [src/deformer.py]
- [MODIFY] [NonLinearParallaxDeformer](file:///e:/Live2DProject/src/deformer.py)
    -   Update `get_deformed_vertices(yaw, pitch)` signature.
    -   Update `get_deformed_point(point, yaw, pitch)` signature.
    -   Update `_deform(centered_pts, weights, yaw, pitch)` logic:
        -   Add `dy = pitch * weights * sensitivity`.
        -   Add vertical compression scaling: `ScaleY = 1.0 - (pitch * strength * y_norm)`.

### [src/live2d.py]
- [MODIFY] [Live2DPart.update](file:///e:/Live2DProject/src/live2d.py)
    -   Extract `Pitch` from tracker data.
    -   Normalize `Pitch` (-1.5 to 1.5).
    -   Pass `normalized_pitch` to `deformer.get_deformed_vertices` and `get_deformed_point`.

## Verification Plan
### Manual Verification
1.  Run `uv run mainArcade.py`.
2.  Perform physical head movements:
    -   Look Up: Face center should move Up. Top of face (forehead) should appear slightly squashed (foreshortened). Bottom (chin) expanded.
    -   Look Down: Face center should move Down. Forehead expanded. Chin squashed.
