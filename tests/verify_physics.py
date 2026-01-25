import sys
import unittest
from unittest.mock import MagicMock
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

# Mock arcade before importing modules that use it
arcade_mock = MagicMock()


class MockSprite:
    pass


arcade_mock.Sprite = MockSprite
arcade_mock.SpriteList = lambda: MagicMock()
sys.modules["arcade"] = arcade_mock
sys.modules["arcade.gl"] = MagicMock()

# Mock Const
const_mock = MagicMock()
const_mock.MODEL_PATH = "assets"
const_mock.GLOBAL_SCALE = 1.0
const_mock.MODEL_DATA = {
    "BackHair": {
        "filename": ["backhair.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
    "FrontHairLeft": {
        "filename": ["front_l.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
    "FrontHairMiddle": {
        "filename": ["front_m.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
    "Body": {"filename": ["body.png"], "global_center_x": 0, "global_center_y": 0},
    "Face": {"filename": ["face.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeWhiteL": {
        "filename": ["eye.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },  # Add missing parts to avoid key error if loop hits them
    "EyeWhiteR": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyePupilL": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyePupilR": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "FaceLandmark": {
        "filename": ["eye.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
    "EyeLidL": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeLidR": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeLashL": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeLashR": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeBrowL": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "EyeBrowR": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "Mouth": {"filename": ["eye.png"], "global_center_x": 0, "global_center_y": 0},
    "FrontHairShadowLeft": {
        "filename": ["eye.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
    "FrontHairShadowMiddle": {
        "filename": ["eye.png"],
        "global_center_x": 0,
        "global_center_y": 0,
    },
}
sys.modules["Const"] = const_mock

# Mock create_mesh
create_mesh_mock = MagicMock()
create_mesh_mock.create_image_mesh = MagicMock(
    return_value=(MagicMock(), MagicMock())
)  # returns pts, tris
# Mock PBDCloth2D to return an object with real numpy arrays
import numpy as np


def mock_pbd(pts, tris, stiffness=0.1, fixed_ratio=0.1):
    m = MagicMock()
    m.pos = np.zeros((10, 2))  # Real numpy array
    m.fixed_indices = np.array([0, 1])
    m.step = MagicMock()
    return m


create_mesh_mock.PBDCloth2D = MagicMock(side_effect=mock_pbd)
sys.modules["create_mesh"] = create_mesh_mock

# Mock mesh_renderer
mesh_renderer_mock = MagicMock()
mesh_renderer_mock.CustomMesh = MagicMock(
    side_effect=lambda ctx, p, pts, tris, scale: MagicMock(
        original_vertices=MagicMock(reshape=lambda *args: MagicMock())
    )
)
mesh_renderer_mock.GridMesh = MagicMock()
sys.modules["mesh_renderer"] = mesh_renderer_mock

# Now import live2d
# We just map src.live2d if needed, but since we modify sys.path, basic import live2d works if we are in right dir?
# But user has file in e:\Live2DProject\src\live2d.py
# We should probably run this script from project root or src.
# Let's assume we run python tests/verify_physics.py
import live2d


class TestLive2DIntegration(unittest.TestCase):
    def test_func_table_structure(self):
        """Verify func_table has 4-element tuples for physics parts"""
        self.assertEqual(len(live2d.func_table["BackHair"]), 4)
        config = live2d.func_table["BackHair"][3]
        self.assertIsInstance(config, dict)
        self.assertEqual(config["type"], live2d.CFG_PHYSICS)

    def test_create_live2dpart_physics(self):
        """Verify physics solver is created for BackHair with correct params"""
        ctx = MagicMock()

        lives, root = live2d.create_live2dpart(ctx)

        back_hair = lives.get("BackHair")
        self.assertIsNotNone(back_hair)
        self.assertTrue(hasattr(back_hair, "physics_solver"))
        self.assertIsNotNone(back_hair.physics_solver)

        # Verify parameters passed to PBDCloth2D
        # BackHair config in func_table: stiffness=0.2, fixed_ratio=0.1
        found = False
        for call in create_mesh_mock.PBDCloth2D.call_args_list:
            kwargs = call.kwargs
            if kwargs.get("stiffness") == 0.2 and kwargs.get("fixed_ratio") == 0.1:
                found = True
                break
        self.assertTrue(
            found, "PBDCloth2D was not called with stiffness=0.2 and fixed_ratio=0.1"
        )

        body = lives.get("Body")
        self.assertIsNone(body.physics_solver)

        # Test Update calls physics step
        back_hair.physics_solver.step = MagicMock()
        back_hair.update(None)
        back_hair.physics_solver.step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
