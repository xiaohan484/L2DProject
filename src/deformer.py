import numpy as np


class NonLinearParallaxDeformer:
    def __init__(self, pts, radius_scale=0.75):
        """
        :param pts: Initial vertices (N, 2)
        """
        self.original_pts = pts.copy()

        # 1. Calculate Center and recenter points
        min_vals = pts.min(axis=0)
        max_vals = pts.max(axis=0)
        self.center = (min_vals + max_vals) / 2.0

        # Centered Points
        self.centered_pts = self.original_pts - self.center

        # 2. Calculate Radius
        width = max_vals[0] - min_vals[0]
        # Use a slightly loose radius to ensure edges don't clamp too abruptly
        self.radius = (width / 2.0) * 1.2
        if self.radius == 0:
            self.radius = 1.0  # Safety

        self.sensitivity = 60.0  # Adjusted from 80.0
        self.compression_strength = 0.4  # For asymmetric squash (perspective)

        # 3. Calculate Weights (Z-Weight) - Precalculated for static mesh
        xs = self.centered_pts[:, 0]
        ys = self.centered_pts[:, 1]

        dists = np.sqrt(xs**2 + ys**2)
        norm_dists = dists / self.radius
        norm_dists = np.clip(norm_dists, 0.0, 1.0)

        # Weight Function: Plateau Curve (1 - r^3)
        # Keeps center rigid, pushes distortion to edges
        self.weights = 1.0 - (norm_dists**3.0)

    def get_deformed_vertices(self, yaw, pitch=0.0):
        return self._deform(self.centered_pts, self.weights, yaw, pitch)

    def get_deformed_point(self, point, yaw, pitch=0.0):
        """
        Transform a single 2D point (e.g. child anchor)
        :param point: [x, y] or [[x, y]]
        """
        p = np.array(point, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, 2)

        # Center displacement logic
        p_centered = p - self.center

        # Calculate dynamic weights for this point
        xs = p_centered[:, 0]
        ys = p_centered[:, 1]
        dists = np.sqrt(xs**2 + ys**2)
        norm_dists = dists / self.radius
        norm_dists = np.clip(norm_dists, 0.0, 1.0)

        weights = 1.0 - (norm_dists**3.0)

        offsets = self._deform(p_centered, weights, yaw, pitch)

        # _deform returns Absolute Coordinates (it adds self.center at the end)
        return offsets[0]

    def _deform(self, centered_pts, weights, yaw, pitch):
        # 1. Displacement (Parallax Bulge)
        # dx = Yaw * Weight * Sensitivity
        pitch = -pitch
        dx = yaw * weights * self.sensitivity
        dy = pitch * weights * self.sensitivity

        # 2. Asymmetric Compression (Perspective Trick)
        # s = 1.0 - Yaw * Strength * (x / R)
        # Normalize X (-1 to 1) using radius
        xs = centered_pts[:, 0]
        ys = centered_pts[:, 1]

        x_norm_signed = xs / self.radius
        x_norm_signed = np.clip(x_norm_signed, -1.0, 1.0)

        y_norm_signed = ys / self.radius
        y_norm_signed = np.clip(y_norm_signed, -1.0, 1.0)

        scale_x = 1.0 - (yaw * self.compression_strength * x_norm_signed)
        # Changed to + because:
        # Look Down (Pitch > 0). Chin (Y < 0).
        # We want Squash (Scale < 1).
        # 1.0 + (Pos * Str * Neg) = 1.0 - (Pos) < 1.0. Correct.
        scale_y = 1.0 + (pitch * self.compression_strength * y_norm_signed)

        # Apply Logic:
        # Move (Bulge) -> Scale (Compress Far Side)
        temp_xs = xs + dx
        new_xs = temp_xs * scale_x

        temp_ys = ys + dy
        new_ys = temp_ys * scale_y

        # Restore Center
        final_pts = np.column_stack((new_xs, new_ys))
        final_pts += self.center

        return final_pts
