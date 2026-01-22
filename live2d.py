from __future__ import annotations
from typing import Optional
import arcade
import numpy as np
from mesh_renderer import GridMesh


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


front_shadow_z = 3
face_feature_z = 2
face_base_z = 1
body_base_z = 0
back_hair_z = -1

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
        elif isinstance(self.views, GridMesh):
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
