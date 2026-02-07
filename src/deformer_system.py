from abc import ABC, abstractmethod
import numpy as np
from parallax_deformer import NonLinearParallaxDeformer


class BaseDeformer(ABC):
    """Abstract base class for all deformers."""

    @abstractmethod
    def transform(self, vertices: np.ndarray, params: dict) -> np.ndarray:
        """
        Transform vertices based on parameters.
        :param vertices: (N, 2) array of vertices.
        :param params: Dictionary of parameters (e.g. {"Yaw": 0, "Pitch": 0})
        :return: Transformed (N, 2) vertices.
        """
        pass

    def transform_and_scale(
        self, point: np.ndarray, params: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a single point (or N points) and return the local scaling.
        Default implementation uses Finite Difference.
        :return: (transformed_points, scales) where scales is (N, 2)
        """
        # Ensure input is (N, 2)
        pts = np.atleast_2d(point)

        # 1. Transform original
        t0 = self.transform(pts, params)

        # 2. Probe X (+epsilon)
        epsilon = 0.1
        pts_x = pts + [epsilon, 0]
        tx = self.transform(pts_x, params)

        # 3. Probe Y (+epsilon)
        pts_y = pts + [0, epsilon]
        ty = self.transform(pts_y, params)

        # 4. Calculate Scale
        # dx_out / dx_in = dist(tx, t0) / epsilon
        # We need component-wise scale? Or magnitude?
        # Typically sx = (tx.x - t0.x) / eps if grid aligned.
        # But if rotated, magnitude is safer.
        # Live2D logic typically uses aligned scaling for simple sprites.

        sx = (tx[:, 0] - t0[:, 0]) / epsilon
        sy = (ty[:, 1] - t0[:, 1]) / epsilon

        scales = np.column_stack((sx, sy))
        return t0, scales


class PerspectiveDeformer(BaseDeformer):
    """
    Wraps NonLinearParallaxDeformer to adapt to the new interface.
    Handles Head Rotation (Yaw/Pitch).
    """

    def __init__(self, original_vertices: np.ndarray, radius_scale=1.5, center=(0, 0)):
        # Initialize the legacy deformer
        # Note: NonLinearParallaxDeformer copies the vertices.
        self.impl = NonLinearParallaxDeformer(original_vertices, radius_scale, center)
        self.deform_ratio_yaw = 1.0
        self.deform_ratio_pitch = 1.0

    def _get_eff_params(self, params: dict):
        # 1. Extract Params using keys expected by the system
        yaw = params.get("Yaw", 0.0)
        pitch = params.get("Pitch", 0.0)

        # 2. Normalize inputs (copied from Live2DPart logic)
        # TODO: Should normalization happen here or in Live2DPart?
        # Ideally, params passed to Deformer are already "Effect Values" or "Raw Values".
        # If we move normalization here, we duplicate logic from Live2DPart.
        # Let's assume params are RAW degrees for now, and we normalize here?
        # OR, Live2DPart passes "NormalizedYaw" in params.

        # Let's keep logic close to legacy for now: Re-implement normalization or expect pre-normalized?
        # The existing `NonLinearParallaxDeformer.get_deformed_vertices` expects normalized values (~ -1.5 to 1.5).

        # We will assume params contains "NormalizedYaw" etc for now?
        # No, Live2DPart.update calculates it.
        # Let's extract "HeadYaw", "HeadPitch" from params which might be the normalized ones.
        # Or standard "Yaw", "Pitch" and we normalize here.

        # Let's perform normalization HERE to encapsulate logic.
        # Input Yaw/Pitch are in degrees.

        normalized_yaw = np.clip(yaw / 45.0, -1.5, 1.5)
        normalized_pitch = np.clip(pitch / 45.0, -1.5, 1.5)

        eff_yaw = normalized_yaw * self.deform_ratio_yaw
        eff_pitch = normalized_pitch * self.deform_ratio_pitch
        return eff_yaw, eff_pitch

    def transform(self, vertices: np.ndarray, params: dict) -> np.ndarray:
        eff_yaw, eff_pitch = self._get_eff_params(params)

        # 3. Transform
        # Problem: NonLinearParallaxDeformer operates on its INTERNAL `centered_pts`.
        # It does NOT properly transform ARBITRARY vertices passed in `transform` method
        # unless we use `transform_points`.
        # `get_deformed_vertices` uses `self.centered_pts`.

        # If `vertices` matches `self.impl.original_pts`, we can use `get_deformed_vertices`.
        # But in a stack, `vertices` might have been modified by previous deformer.
        # So we MUST use `transform_points`.

        new_verts, _ = self.impl.transform_points(vertices, eff_yaw, eff_pitch)

        # Note: transform_points returns (N, 2)
        return new_verts

    def transform_and_scale(
        self, point: np.ndarray, params: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        eff_yaw, eff_pitch = self._get_eff_params(params)
        return self.impl.transform_points(point, eff_yaw, eff_pitch)


class ExpressionDeformer(BaseDeformer):
    """
    Placeholder for Shape Deformation (Smiling, Brows, etc).
    """

    def __init__(self):
        pass

    def transform(self, vertices: np.ndarray, params: dict) -> np.ndarray:
        # Pass-through for now
        return vertices
