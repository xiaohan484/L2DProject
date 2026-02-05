import numpy as np


class NonLinearParallaxDeformer:
    def __init__(self, pts, radius_scale=0.75, center=None):
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
        height = max_vals[1] - min_vals[1]

        # Use the larger dimension to ensure coverage (Face is often taller than wide)
        max_dim = max(width, height)
        self.radius = (max_dim / 2.0) * 1.2

        if self.radius == 0:
            self.radius = 1.0  # Safety

        self.sensitivity = 60.0  # Reduced to keep face "centered" and avoid gaps
        self.compression_strength = 0.4  # For asymmetric squash (perspective)
        self.vis_param_power = 3.0

        # 3. Calculate Weights (Z-Weight) - Precalculated for static mesh
        xs = self.centered_pts[:, 0]
        ys = self.centered_pts[:, 1]

        dists = np.sqrt(xs**2 + ys**2)
        norm_dists = dists / self.radius
        norm_dists = np.clip(norm_dists, 0.0, 1.0)

        # Weight Function: Plateau Curve (1 - r^3)
        # Keeps center rigid, pushes distortion to edges
        self.weights = 1.0 - (norm_dists**self.vis_param_power)

    def get_deformed_vertices(self, yaw, pitch=0.0):
        # Yaw/Pitch in range roughly [-1.5, 1.5]
        pts, _, _ = self._deform(self.centered_pts, self.weights, yaw, pitch)
        return pts

    def get_deformed_point(self, point, yaw, pitch=0.0):
        """
        Transform a single 2D point (e.g. child anchor)
        :param point: [x, y] or [[x, y]]
        """
        p = np.array(point, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, 2)

        # We assume 'point' is in Global/World coordinates (or rather, the same space as 'pts' passed to __init__)
        # But wait, 'pts' passed to __init__ were usually "Render Vertices" which are relative to the Sprite Center?
        # In live2d.py, we initiated it with CustomMesh.original_vertices.
        # CustomMesh vertices are usually centered around the sprite center.
        #
        # Accessing child.x, child.y gives Local Coordinates relative to Parent Center?
        # In live2d.py:
        # eff_x = self.x (Local to Parent)
        #
        # If Parent is Face, Face Center is (0,0) in Local Space?
        # Let's check live2d.py hierarchy.
        # Live2DPart: x, y are loaded as "Global Position" then converted to "Local Position".
        # If Parent is Body (at ~600, 1200), Face (at ~600, 400).
        # Face Local = (0, -800) relative to Body.
        #
        # Deformer is initialized with `view.original_vertices`.
        # `view` (CustomMesh) usually centers the mesh so mean is (0,0)?
        # No, CustomMesh usually keeps vertices relative to image center or something.
        #
        # Key: `NonLinearParallaxDeformer` calculates `self.center` from `pts`.
        # So it knows where the "Mesh Center" is in the provided coordinate space.
        #
        # `get_deformed_point` expects `point` in the SAME coordinate space as `pts`.
        #
        # In `live2d.py`, we will pass `eff_x, eff_y`. These are local coordinates of the Child relative to Parent.
        # Are `pts` (render verts) also in Parent Local Space?
        # `render_verts` comes from `CustomMesh`.
        # CustomMesh `original_vertices` are typically centered if `create_image_mesh` was used and centered.
        # But `create_image_mesh` usually returns image coordinates.
        # CustomMesh constructor wraps them.
        # If `CustomMesh` doesn't re-center, they are image coords.
        #
        # Let's assume `pts` and `imp` inputs are consistent (both Local or both Global).
        # Since we use `eff_x` (Local), we should ensure `deformer` works in Local.

        return self.transform_points(p, yaw, pitch)[0]

    def transform_points(self, points, yaw, pitch=0.0):
        """
        Deform arbitrary points using this deformer's field.
        Points should be in the same coordinate space as the initialized mesh.
        Returns:
            deformed_points: (N, 2)
            scales: (N, 2)  [scale_x, scale_y] for each point
        """
        pts = np.array(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # 1. Center them (using Face Center)
        # pts are in Anchor Space (same as self.original_pts)
        # We need them in Center Space for deformation logic
        centered_p = pts - self.center

        # 2. Calculate Weights dynamically
        xs = centered_p[:, 0]
        ys = centered_p[:, 1]
        dists = np.sqrt(xs**2 + ys**2)
        norm_dists = dists / self.radius
        norm_dists = np.clip(norm_dists, 0.0, 1.0)

        weights = 1.0 - (norm_dists**self.vis_param_power)

        # 3. Deform
        final_pts, scale_x, scale_y = self._deform(centered_p, weights, yaw, pitch)

        scales = np.column_stack((scale_x, scale_y))

        return final_pts, scales

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

        # Yaw < 0 (Look Left) -> Left Side (x < 0) -> Compress (Scale < 1)
        # 1 - (-1) * 0.4 * (-1) = 1 - 0.4 = 0.6. Correct.
        scale_x = 1.0 - (yaw * self.compression_strength * x_norm_signed)

        # Pitch > 0 (Look Down) -> Chin (y > 0 if Y down, or y < 0 if Y up? Ref: live2d.py uses y down?)
        # In this script, we assume coordinate consistency.
        # If we use `scale_y = 1.0 + ...`, let's verify.
        scale_y = 1.0 - (pitch * self.compression_strength * y_norm_signed)

        # Apply Logic:
        # Move (Bulge) -> Scale (Compress Far Side)
        temp_xs = xs + dx
        new_xs = temp_xs * scale_x

        temp_ys = ys + dy
        new_ys = temp_ys * scale_y

        # Restore Center
        final_pts = np.column_stack((new_xs, new_ys))
        final_pts += self.center

        return final_pts, scale_x, scale_y
