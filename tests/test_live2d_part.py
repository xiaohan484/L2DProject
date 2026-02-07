import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

# Mock arcade before importing live2d
sys.modules["arcade"] = MagicMock()
sys.modules["arcade.gl"] = MagicMock()
sys.modules["mesh_renderer"] = MagicMock()
sys.modules["create_mesh"] = MagicMock()


# Fix isinstance(view, arcade.Sprite) check
class MockSprite:
    pass


sys.modules["arcade"].Sprite = MockSprite
sys.modules["arcade"].SpriteList = MagicMock

import live2d

import live2d
from live2d import Live2DPart


class TestLive2DPart(unittest.TestCase):
    def setUp(self):
        # Create a simple parent-child hierarchy
        self.parent = Live2DPart(name="Parent", x=100, y=100)
        self.child = Live2DPart(name="Child", x=50, y=0, parent=self.parent)

        # Mock views to avoid crash in update
        self.parent.views = MagicMock()
        self.child.views = MagicMock()

    def test_hierarchy_transform(self):
        """Test that child follows parent's position."""
        # Initial update
        self.parent.update(None)

        # Parent World Pos should be (100, 100)
        p_x, p_y = self.parent.get_world_position()
        self.assertEqual(p_x, 100)
        self.assertEqual(p_y, 100)

        # Child Local Pos is (50, 0) relative to parent
        # Child World Pos should be (150, 100)
        c_x, c_y = self.child.get_world_position()
        self.assertEqual(c_x, 150)
        self.assertEqual(c_y, 100)

    def test_hierarchy_rotation(self):
        """Test that child rotates around parent."""
        # Rotate parent 90 degrees
        self.parent.angle = 90
        self.parent.update(None)

        # Child was at (50, 0) relative.
        # Rotated 90 deg: (0, 50) relative.
        # Parent at (100, 100).
        # Child World: (100, 150).

        c_x, c_y = self.child.get_world_position()
        self.assertAlmostEqual(c_x, 100)
        self.assertAlmostEqual(c_y, 150)

    def test_update_calls_children(self):
        """Test that updating parent recursively updates children."""
        self.child.update = MagicMock()
        self.parent.update(None)
        self.child.update.assert_called_once()

    def test_deformer_stack(self):
        """Test that deformers in the stack are applied."""
        # Create a mock deformer
        mock_deformer = MagicMock()

        # It should return modified vertices
        # Helper to simply add (10, 10)
        def transform_side_effect(verts, params):
            return verts + 10.0

        def transform_and_scale_side_effect(point, params):
            return point + 10.0, np.array([[1.0, 1.0]] * len(point))

        mock_deformer.transform.side_effect = transform_side_effect
        mock_deformer.transform_and_scale.side_effect = transform_and_scale_side_effect

        # Assign to parent
        self.parent.deformers = [mock_deformer]

        # Parent View needs original_vertices
        self.parent.views.original_vertices = np.zeros((1, 4))  # x,y,z,w
        self.parent.views.current_vertices = np.zeros((1, 4))
        self.parent.views.update_buffer = MagicMock()

        # Update
        self.parent.update({"Yaw": 0, "Pitch": 0})

        # Check if transform was called (Parent Mesh + Child Mesh Binding)
        # Should be called at least once
        self.assertTrue(mock_deformer.transform.called)

        # Check if buffer updated (0 + 10 = 10)
        # current_vertices flattened
        updated_buffer = self.parent.views.current_vertices.reshape(-1, 4)
        self.assertEqual(updated_buffer[0, 0], 10.0)
        self.assertEqual(updated_buffer[0, 1], 10.0)


if __name__ == "__main__":
    unittest.main()
