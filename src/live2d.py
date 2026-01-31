from __future__ import annotations
from typing import Optional
import arcade
import numpy as np
from mesh_renderer import GridMesh

try:
    from mesh_renderer import CustomMesh
except ImportError:
    pass
from create_mesh import create_image_mesh, PBDCloth2D
from Const import MODEL_PATH, MODEL_DATA, GLOBAL_SCALE
from response import *
from collections import OrderedDict
import os
import time

front_shadow_z = 3
face_feature_z = 2
face_base_z = 1
body_base_z = 0
back_hair_z = -1

# Config types
CFG_GRID = 0
CFG_PHYSICS = 1

PART_HIERARCHY = OrderedDict(
    {
        "Body": (body_base_z, body_response, None, CFG_GRID),
        "Face": (face_base_z, face_response, "Body", CFG_GRID),
        "BackHair": (
            back_hair_z,
            None,
            "Face",
            {
                "type": CFG_PHYSICS,
                "stiffness": 0.3,
                "fixed_ratio": 0.4,
                "damping": 0.99,
                "lra_stiffness": 0.5,
                "wind_strength": 10.0,
            },
        ),
        "EyeWhiteL": (face_feature_z, white_response_l, "Face", CFG_GRID),
        "EyeWhiteR": (face_feature_z, white_response_r, "Face", CFG_GRID),
        "EyePupilL": (face_feature_z, pupils_response_l, "Face", CFG_GRID),
        "EyePupilR": (face_feature_z, pupils_response_r, "Face", CFG_GRID),
        "FaceLandmark": (face_feature_z, None, "Face", CFG_GRID),
        "EyeLidL": (face_feature_z, lid_response_l, "Face", CFG_GRID),
        "EyeLidR": (face_feature_z, lid_response_r, "Face", CFG_GRID),
        "EyeLashL": (face_feature_z, lash_response_l, "Face", CFG_GRID),
        "EyeLashR": (face_feature_z, lash_response_r, "Face", CFG_GRID),
        "EyeBrowL": (face_feature_z, None, "Face", CFG_GRID),
        "EyeBrowR": (face_feature_z, None, "Face", CFG_GRID),
        "Mouth": (face_feature_z, mouth_response, "Face", CFG_GRID),
        # "FrontHairShadowLeft": (front_shadow_z, None, "Face", CFG_GRID),
        # "FrontHairShadowMiddle": (front_shadow_z, None, "Face", CFG_GRID),
        "FrontHairLeft": (
            face_feature_z,
            None,
            "Face",
            {
                "type": CFG_PHYSICS,
                "stiffness": 0.3,
                "fixed_ratio": 0.3,
                "max_area": 500,
                "simplify_epsilon": 1.0,
                "lra_stiffness": 0.8,
                "wind_strength": 20.0,
            },
        ),
        "FrontHairMiddle": (
            face_feature_z,
            None,
            "Face",
            # CFG_GRID,
            {
                "type": CFG_PHYSICS,
                "stiffness": 0.3,
                "fixed_ratio": 0.2,
                "max_area": 500,
                "damping": 0.8,
                "simplify_epsilon": 1.0,
                "lra_stiffness": 1.0,
            },
        ),
    }
)


def load_local_position(global_pos, parent_global_pos, global_scale):
    # Calculate the relative distance in global space
    delta_x = global_pos[0] - parent_global_pos[0]
    delta_y = global_pos[1] - parent_global_pos[1]

    # Scale down the delta to get the original local position
    local_x = delta_x / global_scale
    local_y = delta_y / global_scale

    return local_x, local_y


def create_live2dpart(ctx):
    lives = OrderedDict({})
    for name, args in PART_HIERARCHY.items():
        # Handle 3 or 4 args for robustness, though we updated all to 4
        if len(args) == 4:
            _, _, parent, _ = args
        else:
            _, _, parent = args

        if parent is not None:
            parent = lives[parent]
        lives[name] = create_live2dpart_each(ctx, name, parent)
    root = lives["Body"]
    lives.move_to_end("BackHair", last=False)
    return lives, root


def create_live2dpart_each(ctx, name, parent):
    # Unpack config
    args = PART_HIERARCHY[name]
    z = args[0]
    res = args[1]

    config_type = CFG_GRID
    physics_params = {}

    if len(args) >= 4:
        cfg = args[3]
        if isinstance(cfg, dict):
            config_type = cfg.get("type", CFG_GRID)
            physics_params = cfg
        else:
            config_type = cfg

    data = MODEL_DATA[name]

    raw_global_x = data["global_center_x"]
    raw_global_y = data["global_center_y"]

    view = None
    physics_solver = None

    if config_type == CFG_PHYSICS:
        # PBD Physics Setup
        filename = data["filename"][0]  # Assume 1 file for physics parts for now
        image_path = os.path.join(MODEL_PATH, filename)

        # 1. Generate Mesh with density params
        area = physics_params.get("max_area", 500)
        epsilon = physics_params.get("simplify_epsilon", 1.0)
        pts, tris = create_image_mesh(
            image_path, debug=False, max_area=area, simplify_epsilon=epsilon
        )

        # 2. Create CustomMesh
        view = CustomMesh(ctx, image_path, pts, tris, scale=GLOBAL_SCALE)

        # 3. Initialize Physics
        render_verts = view.original_vertices.reshape(-1, 4)[:, :2]
        sim_pts = render_verts.copy()

        # Extract params with defaults
        stiff = physics_params.get("stiffness", 0.3)
        ratio = physics_params.get("fixed_ratio", 0.1)
        damp = physics_params.get("damping", 0.99)
        lra = physics_params.get("lra_stiffness", 0.8)

        physics_solver = PBDCloth2D(
            sim_pts,
            tris,
            stiffness=stiff,
            fixed_ratio=ratio,
            damping=damp,
            lra_stiffness=lra,
        )
        physics_solver.wind_strength = physics_params.get("wind_strength", 0.0)

    else:
        # Standard GridMesh Setup
        textures = {}
        for path in data["filename"]:
            filepath = os.path.join("assets/sample_model/processed", path)
            p = path.removesuffix(".png")
            textures[p] = filepath

        view = GridMesh(
            ctx,
            textures,
            grid_size=(10, 10),
            scale=GLOBAL_SCALE,
            parent=None,
            data_key=data,  # 10x10 的格子
        )

    if parent == None:
        return Live2DPart(
            name=name,
            parent=None,
            x=0,
            y=0,
            z=z,
            scale_x=GLOBAL_SCALE,
            scale_y=GLOBAL_SCALE,
            angle=0,
            view=view,
            response=res,
            physics_solver=physics_solver,
        )
    else:
        parent_global = (
            MODEL_DATA[parent.name]["global_center_x"],
            MODEL_DATA[parent.name]["global_center_y"],
        )
        pos = load_local_position(
            global_pos=(raw_global_x, raw_global_y),
            parent_global_pos=parent_global,
            global_scale=GLOBAL_SCALE,
        )
        return Live2DPart(
            name=name,
            parent=parent,
            x=pos[0],
            y=-pos[1],
            z=z,
            scale_x=GLOBAL_SCALE,
            scale_y=GLOBAL_SCALE,
            angle=0,
            view=view,
            response=res,
            physics_solver=physics_solver,
        )


def get_local_matrix(angle, sx, sy, tx, ty):
    rad = np.radians(angle)
    cos = np.cos(rad)
    sin = np.sin(rad)
    return np.array(
        [
            [sx * cos, -sy * sin, tx],
            [sx * sin, sy * cos, ty],
            [0, 0, 1],
        ]
    )


PARALLAX_X_STRENGTH = -0.4
PARALLAX_Y_STRENGTH = 0.5

# Body is zero

# Face is Base of other feature
OFFSET_MID = 1.0  # 臉型
OFFSET_BACK = 0.9999  # 後髮
OFFSET_FRONT = 1.0001  # 前髮、五官
OFFSET_FRONT_SHADOW = 0.5

rotate_response = {
    front_shadow_z: (
        PARALLAX_X_STRENGTH * OFFSET_FRONT_SHADOW,
        4 * PARALLAX_Y_STRENGTH * OFFSET_FRONT_SHADOW,
    ),
    face_feature_z: (
        PARALLAX_X_STRENGTH * OFFSET_FRONT,
        PARALLAX_Y_STRENGTH * OFFSET_FRONT,
    ),
    face_base_z: (PARALLAX_X_STRENGTH * OFFSET_MID, PARALLAX_Y_STRENGTH * OFFSET_MID),
    body_base_z: (0, 0),
    back_hair_z: (
        PARALLAX_X_STRENGTH * OFFSET_BACK,
        PARALLAX_Y_STRENGTH * OFFSET_BACK,
    ),
}


class Live2DPart:
    """
    This class handles pure logic and data.
    It doesn't know about Arcade.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Live2DPart] = None,
        x: float = 0,
        y: float = 0,
        z: int = 0,
        angle: float = 0,
        scale_x: float = 1,
        scale_y: float = 1,
        view=None,
        response=None,
        physics_solver=None,
    ):
        self.name = name
        # Local space properties
        self.z_depth = z  # For parallax

        self.angle = angle
        self.sx = scale_x
        self.sy = scale_y
        self.x = x
        self.y = y
        self.parent = parent
        self.response = response
        self.physics_solver = physics_solver
        self.physics_initialized = False
        if self.physics_solver is not None:
            self.physics_init_local_pos = self.physics_solver.pos.copy()
        else:
            self.physics_init_local_pos = None

        if parent:
            parent.set_child(self)
        self.local_matrix = np.eye(3)
        self.world_matrix = np.eye(3)
        self.children = []
        # The visual representation (Composition)
        if isinstance(view, arcade.Sprite):
            self.views = arcade.SpriteList()
            self.views.append(view)
        else:
            self.views = view

        self.add_x = 0
        self.add_y = 0
        self.add_angle = 0
        self.update()

    def set_child(self, child: Live2DPart):
        """
        set child to the Part
        :param child: The Live2DPart which to be child
        :type child: Live2DPart
        """
        self.children.append(child)
        return

    def update(self, data=None):
        """
        update Matrix and recursively update children
        """
        # self.response()
        (offset_x, offset_y) = (0, 0)
        skip = data is None or self.response is None
        if skip is False:
            self.response(self, data)
        if data is not None:
            (offset_x, offset_y) = rotate_response[self.z_depth]
            yaw = data["Yaw"]
            pitch = data["Pitch"]
            offset_x *= yaw
            offset_y *= -pitch
        self.local_matrix = get_local_matrix(
            self.angle,
            self.sx,
            self.sy,
            self.x + self.add_x + offset_x,
            self.y + self.add_y + offset_y,
        )
        if self.parent:
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            self.world_matrix = self.local_matrix

        # Physics Update
        if self.physics_solver is not None:
            # Physics runs in WORLD SPACE to handle gravity and inertia correctly.

            # 1. Initialize logic (First Frame or Reset)
            if not self.physics_initialized:
                # Transform local rest positions to world space so sim starts at correct location
                pts = self.physics_solver.pos
                ones = np.ones((len(pts), 1))
                pts_h = np.hstack([pts, ones])

                # World = Mat @ Local
                # (3, 3) @ (3, N) -> (3, N) -> T -> (N, 3)
                world_pts = (self.world_matrix @ pts_h.T).T
                self.physics_solver.pos = world_pts[:, :2]
                self.physics_solver.prev_pos = (
                    self.physics_solver.pos.copy()
                )  # Reset velocity
                self.physics_initialized = True

            # 2. Update Fixed Points (Anchors) to current World Position
            # We want the anchors to strictly follow the head (parent)
            fixed_idx = self.physics_solver.fixed_indices

            # Get their original local positions
            local_fixed = self.physics_init_local_pos[fixed_idx]

            # Transform to World
            ones = np.ones((len(local_fixed), 1))
            local_fixed_h = np.hstack([local_fixed, ones])
            # Matrix Mult
            world_fixed = (self.world_matrix @ local_fixed_h.T).T

            # Force solver positions
            self.physics_solver.pos[fixed_idx] = world_fixed[:, :2]

            # 3. Step Physics
            # Pass wind params if defined in self.physics_solver (not stored there directly but can be config derived)
            # Ideally we store config in Live2DPart or Solver.
            # For now, let's just use a default or hack it.
            # In real system, we should have stored 'wind_strength' in the solver object or Part config.
            # Let's attach 'wind_strength' to the solver object in create logic or Config.

            wind = getattr(self.physics_solver, "wind_strength", 0.0)
            self.physics_solver.step(10, wind_strength=wind, time_val=time.time())

            # 4. Prepare for Rendering (World -> Local)
            # The View (CustomMesh) expects LOCAL coordinates because it is drawn with self.world_matrix.
            try:
                inv_mat = np.linalg.inv(self.world_matrix)
            except np.linalg.LinAlgError:
                inv_mat = np.eye(3)

            sim_world = self.physics_solver.pos
            ones = np.ones((len(sim_world), 1))
            sim_world_h = np.hstack([sim_world, ones])

            sim_local = (inv_mat @ sim_world_h.T).T

            # 5. Update Mesh VBO
            if hasattr(self.views, "current_vertices"):
                data_view = self.views.current_vertices.reshape(-1, 4)
                data_view[:, :2] = sim_local[:, :2]
                self.views.update_buffer()

        for child in self.children:
            child.update(data)

    def sync(self):
        """
        The only moment we interact with other coordinate system.
        """
        if isinstance(self.views, arcade.SpriteList):
            self.views[0].center_x = self.world_matrix[0, 2]
            self.views[0].center_y = self.world_matrix[1, 2]
            self.views[0].scale_x = self.sx
            self.views[0].scale_x = self.sx
            self.views[0].angle = self.angle
            # self.views.draw(self.world_matrix)
            # self.views.center_x = self.world_matrix[0, 2]
            # self.views.center_y = self.world_matrix[1, 2]
            # self.views.scale_x = self.sx
            # self.views.scale_y = self.sy
            # self.views.angle = self.angle

    def draw(self, **kwargs):
        if isinstance(self.views, arcade.SpriteList):
            self.sync()
            self.views.draw(**kwargs)
        elif isinstance(self.views, GridMesh) or isinstance(self.views, CustomMesh):
            self.views.draw(self.world_matrix)

    def get_world_position(self):
        """
        from world matrix last column
        """
        return self.world_matrix[0, 2], self.world_matrix[1, 2]


if __name__ == "__main__":
    # === 測試場景 ===
    # 建立一個手臂結構：肩膀 (Root) -> 手肘 (Child) -> 手掌 (Grandchild)
    print("--- 初始化骨骼 ---")
    # 肩膀在世界座標 (100, 100)
    shoulder = Live2DPart("Shoulder", x=100, y=100)
    # 手肘在肩膀的右邊 50 單位 (相對座標 50, 0)
    elbow = Live2DPart("Elbow", x=50, y=0, parent=shoulder)
    # 手掌在手肘的右邊 30 單位 (相對座標 30, 0)
    hand = Live2DPart("Hand", x=30, y=0, parent=elbow)
    # 強制更新一次矩陣
    shoulder.update()
    print(f"肩膀世界座標: {shoulder.get_world_position()} (預期: 100, 100)")
    print(f"手肘世界座標: {elbow.get_world_position()}    (預期: 150, 100)")
    print(f"手掌世界座標: {hand.get_world_position()}     (預期: 180, 100)")
    print("\n--- 測試旋轉 (FK) ---")
    # 旋轉肩膀 90 度 (整隻手臂應該垂直向下指，因為螢幕座標 Y 向下通常是正的，或向上視你的系通而定)
    # 假設 Y 向下為正 (Arcade 預設 Y 向上，這裡我們先看數值)
    shoulder.angle = 90
    shoulder.update()
    # 數學預測：
    # 肩膀還是在 (100, 100)
    # 手肘應該轉到肩膀「下面」 (100, 150) (如果是 Y 軸向下增加) 或是 (100, 50) (如果 Y 向上增加)
    # 手掌應該在手肘「下面」
    wx, wy = hand.get_world_position()
    print(f"肩膀旋轉 90 度後，手掌的世界座標: ({wx:.2f}, {wy:.2f})")
    if abs(wx - 100) < 0.1:
        print(">> PASS: 手掌正確跟隨肩膀旋轉！")
    else:
        print(">> FAIL: 骨骼連動失效，請檢查矩陣乘法。")
