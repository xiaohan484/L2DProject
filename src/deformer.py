import numpy as np


class NonLinearParallaxDeformer:
    def __init__(self, pts, radius_scale=0.6, center=None):
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

        # Use the larger dimension to ensure coverage
        max_dim = max(width, height)
        # Reduce radius to increase curvature and reduce "swing" distance
        # 0.5 would be a sphere fitting exactly. 0.6 gives a bit of slack.
        # self.radius = (
        #     (max_dim / 2.0) * 1.2 * radius_scale
        # )  # ~0.72 of half-dim if scale is 0.6?
        # Previously: (max_dim/2)*1.2 (default scale=1.0? No default was 0.75 in call?)
        # Let's just set it relative to half-dim clearly.
        # self.radius = (
        #     max_dim / 2.0
        # ) * 1.1  # 1.1x Half-Width. Closer to "Real" head size.
        # MATCH TEST: pitch_test.py uses max_dim * 0.75.
        # (max_dim / 2.0) * 1.5 = max_dim * 0.75.
        self.radius = (max_dim / 2.0) * 1.5

        if self.radius == 0:
            self.radius = 1.0  # Safety

        self.sensitivity = 60.0  # This affects YAW bulge.
        self.compression_strength = 0.5  # Increased slightly for more depth
        self.vis_param_power = 3.0

        # 3. Calculate Weights (Z-Weight)
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
        # CustomMesh constructor wraps them.
        # If `CustomMesh` doesn't re-center, they are image coords.
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
        # --- 1. YAW DEFORMATION (Parallax) ---
        # dx = Yaw * Weight * Sensitivity
        dx = yaw * weights * self.sensitivity

        # Yaw Scale (Perspective)
        xs = centered_pts[:, 0]
        x_norm_signed = xs / self.radius
        x_norm_signed = np.clip(x_norm_signed, -1.0, 1.0)

        scale_x_yaw = 1.0 - (yaw * self.compression_strength * x_norm_signed)

        # Apply Yaw Displacement
        temp_xs = xs + dx

        # --- 2. PITCH DEFORMATION (Cylinder) ---
        # We model the face as being projected onto a Cylinder aligned with X-axis.
        # y = R * sin(phi)
        # We rotate phi by pitch_angle -> phi' = phi + theta
        # y' = R * sin(phi')
        # This naturally compresses edges (Recedes) instead of extending them.

        ys = centered_pts[:, 1]

        # Map y to angle phi
        # y / R = sin(phi)  => phi = arcsin(y/R)
        # We need to clamp y/R to [-1, 1]
        y_norm = ys / self.radius
        y_norm = np.clip(y_norm, -1.0, 1.0)

        # Safe arcsin
        phi = np.arcsin(y_norm)

        # Pitch Angle (Theta)
        # Pitch input is roughly [-1.5, 1.5].
        # let's say 1.0 corresponds to ~30 degrees or similar.
        # We can tune this factor.
        # A simple linear mapping works best for now.
        # FIX: User repored rotation direction was wrong.
        # Reverting to negative coefficient.
        theta = -pitch * 0.4

        phi_new = phi + theta

        # Check for wrap-around or clamping
        # We don't want the texture to wrap around the back of the head visibly if possible,
        # or we accept it. Clamping phi to somewhat inside [-PI/2, PI/2] avoids back-face logic.
        # But allow slight margin.

        # Calculate new Y
        # y' = R * sin(phi')
        # We must maintain the x-scale caused by this Z-depth change?
        # Z = R * cos(phi).
        # New Z' = R * cos(phi').
        # Scale factor due to recession = Z' / Z ? (Perspective)
        # Or simply derivative dy'/dy?

        # Let's simpler first: just Y mapping.
        new_ys = self.radius * np.sin(phi_new)

        # Scale Calculation for Pitch (Vertical Compression)
        # dy'/dy = (dy'/dphi) * (dphi/dy)
        # y = R sin(phi) => dy/dphi = R cos(phi)
        # y' = R sin(phi+th) => dy'/dphi = R cos(phi+th)
        # dy'/dy = cos(phi+th) / cos(phi)

        cos_phi = np.sqrt(1.0 - y_norm**2)  # cos(arcsin(x)) = sqrt(1-x^2)
        cos_phi = np.maximum(cos_phi, 0.01)  # Avoid div zero at absolute edges

        cos_phi_new = np.cos(phi_new)

        # If we rotate too far, cos_phi_new might be negative (back face).
        # We might want to mask them or just let them squash.

        scale_y_cyl = cos_phi_new / cos_phi

        # --- 3. PITCH X-PERSPECTIVE (Trapezoid) ---
        # "Depth" effect:
        # Pitch > 0 (Look Down): Chin (y>0) is further -> Compress X. Forehead (y<0) is closer -> Expand X.
        # Factor depends on Y position (y_norm).
        # We use 'sin(phi_new)' (the new Y projection) or just y_norm?
        # Using z-depth is more physically correct.
        # z = R * cos(phi). z_new = R * cos(phi_new).
        # Scale ~ z_new / z_center? Or relative to projection plane?
        # Simple Trapzoid:
        # ScaleX *= 1 + (Pitch * Strength * -y_norm)
        # Note: y_norm is down.
        # Look Down (Pitch > 0). y_norm > 0 (Chin). We want compress -> Scale < 1.
        # So term should be negative. (1 - Pitch * y).
        # y > 0 -> 1 - pos = < 1. Correct.
        # Look Down (Pitch > 0) -> Theta Negative -> Features move Down.
        # Bottom (y < 0) should Compress (Recede).
        # Top (y > 0) should Expand.
        # FIX: Flip sign to match new rotation direction.
        # Look Down (Pitch > 0) -> Theta Negative -> Features move Down.
        # Bottom (y < 0) should Compress (Recede).
        # --- 3. PITCH X-PERSPECTIVE (Depth Scaling) ---
        # Instead of linear trapezoid (1 + k*y), we use the actual Z-depth change.
        # Z ~ cos(phi). Scale ~ NewZ / OldZ = cos(phi_new) / cos(phi).
        # This matches the Y-foreshortening factor 'scale_y_cyl'.

        # We blend this effect based on 'perspective_strength' (0.0 = Orthographic, 1.0 = Full Perspective)
        perspective_strength = 0.5  # Adjustable. 0.5 is a safe middle ground.

        z_scale = scale_y_cyl  # cos_phi_new / cos_phi

        # Apply scaling relative to 1.0
        scale_x_pitch = 1.0 + (z_scale - 1.0) * perspective_strength

        # Combine X Scales
        scale_x = scale_x_yaw * scale_x_pitch

        # Apply X Transformation
        new_xs = temp_xs * scale_x

        # What about Scale Y?
        # Previously we had `scale_y` for perspective in Pitch.
        # Now `scale_y_cyl` handles it.

        scale_y = scale_y_cyl

        # Restore Center
        final_pts = np.column_stack((new_xs, new_ys))
        final_pts += self.center

        return final_pts, scale_x, scale_y
