import unittest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from parallax_deformer import NonLinearParallaxDeformer


class TestNonLinearParallaxDeformer(unittest.TestCase):
    def setUp(self):
        # Create a simple 100x100 square mesh centered at 0,0 for easier reasoning
        # Typically vertices might be centered or not. Deformer handles centering.
        # Let's define a grid of points: (-50,-50) to (50,50)
        x = np.linspace(-50, 50, 3)  # -50, 0, 50
        y = np.linspace(-50, 50, 3)
        xv, yv = np.meshgrid(x, y)
        self.pts = np.column_stack((xv.ravel(), yv.ravel()))

        # Initialize Deformer
        # Radius scale implied. Max dim = 100. Radius approx 100/2 * 1.5 = 75
        self.deformer = NonLinearParallaxDeformer(self.pts, radius_scale=1.0)

    def test_initialization(self):
        """Verify weights are calculated correctly."""
        # Center point (0,0) should have max weight (1.0 or close)
        # Deformer calculates weights as 1 - (r/R)^3

        # Find index of (0,0)
        center_idx = 4  # Middle of 3x3 grid
        center_pt = self.pts[center_idx]
        self.assertTrue(np.allclose(center_pt, [0, 0]))

        weight = self.deformer.weights[center_idx]
        self.assertAlmostEqual(weight, 1.0, places=5)

        # Edge point (-50, -50)
        # r = sqrt(50^2 + 50^2) = 70.71
        # R = 75 (approx)
        # r/R = 0.94
        # weight = 1 - 0.94^3 = 1 - 0.83 = 0.17
        edge_idx = 0
        weight_edge = self.deformer.weights[edge_idx]
        self.assertLess(weight_edge, 1.0)
        self.assertGreater(weight_edge, 0.0)

    def test_zero_rotation(self):
        """Test that (0,0) rotation returns original points."""
        res_pts = self.deformer.get_deformed_vertices(0.0, 0.0)
        self.assertTrue(
            np.allclose(res_pts, self.pts), "Zero rotation should not change points"
        )

    def test_yaw_deformation(self):
        """Test Yaw (Head Turn Left/Right)."""
        # Yaw > 0 (Turn Right -> Nose moves Right)
        # Bulge X: dx = yaw * weight * sensitivity
        # Perspective X: Left side (x<0) expands? Right side (x>0) compresses?
        # Let's check logic: scale_x = 1.0 - (yaw * str * x_norm)
        # If yaw=1, x_node > 0 (Right side). 1 - pos = < 1 (Compress). Correct.
        # x_node < 0 (Left side). 1 - neg = > 1 (Expand). Correct.

        yaw = 0.5  # 0.5 rad approx 28 deg
        res_pts = self.deformer.get_deformed_vertices(yaw=yaw, pitch=0.0)

        # Check Center Point displacement
        # Center weight ~ 1.0. dx ~ 0.5 * 1.0 * sensitivity(60) = 30
        center_idx = 4
        orig_x = self.pts[center_idx][0]  # 0
        new_x = res_pts[center_idx][0]
        self.assertGreater(new_x, orig_x, "Yaw > 0 should move center to right")

        # Check Right Edge compression
        # Right edge (50, 0) -> Index 5
        # It moves right (bulge) but also compresses (perspective)?
        # Actually Bulge moves it right. Perspective moves it left (relative to center) or scales it down?
        # The logic is X_new = (X + dx) * Scale
        pass

    def test_pitch_deformation(self):
        """Test Pitch (Head Turn Up/Down)."""
        # Pitch > 0 (Look Down)
        # Y coordinates should shift down?
        # Cylinder logic: new_y = R * sin(phi + theta). theta = -pitch * 0.4.
        # Pitch > 0 -> Theta < 0.
        # phi (angle) decreases.
        # y = R sin(phi). phi decreases -> y decreases.
        # So Look Down -> Features move Down.

        pitch = 0.5
        res_pts = self.deformer.get_deformed_vertices(yaw=0.0, pitch=pitch)

        center_idx = 4
        orig_y = self.pts[center_idx][1]  # 0
        new_y = res_pts[center_idx][1]

        self.assertLess(
            new_y, orig_y, "Pitch > 0 (Down) should move features down (Y decrease)"
        )


if __name__ == "__main__":
    unittest.main()
