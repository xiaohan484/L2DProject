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


from Const import GLOBAL_SCALE


def initial_coorLive(part, data):
    raw_global_x = data["global_center_x"]
    raw_global_y = data["global_center_y"]
    if part.parent:
        base_x = raw_global_x - part.parent.initial_global_x
        base_y = raw_global_y - part.parent.initial_global_y
        part.x = base_x
        part.y = base_y
        part.local_x = base_x
        part.local_y = base_y
        part.initial_global_x = raw_global_x
        part.initial_global_y = raw_global_yal_y = raw_global_y
    else:
        part.x = 0
        part.y = 0 - 950 * GLOBAL_SCALE  # 試著改成 +50 或 -50 看看效果
        part.center_x = part.base_local_x
        part.center_y = part.base_local_y
        # 記錄初始絕對座標 (這兩行不變)
        part.initial_global_x = raw_global_x
        part.initial_global_y = raw_global_yal_y = raw_global_y


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
        angle: float = 0,
        scale_x: float = 1,
        scale_y: float = 1,
        view=None,
    ):
        self.name = name
        # Local space properties
        self.z_depth = 1.0  # For parallax

        self.angle = angle
        self.sx = scale_x
        self.sy = scale_y
        self.x = x
        self.y = y
        self.parent = parent
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
        self.update()

    def set_child(self, child: Live2DPart):
        """
        set child to the Part
        :param child: The Live2DPart which to be child
        :type child: Live2DPart
        """
        self.children.append(child)
        return

    def update(self, add_x=0, add_y=0):
        """
        update Matrix and recursively update children
        """
        self.local_matrix = get_local_matrix(
            self.angle, self.sx, self.sy, self.x + add_x, self.y + add_y
        )
        if self.parent:
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            self.world_matrix = self.local_matrix

        for child in self.children:
            child.update()

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
        else:
            self.views.center_x = self.world_matrix[0, 2]
            self.views.center_y = self.world_matrix[1, 2]
            self.views.scale_x = self.sx
            self.views.scale_y = self.sy
            self.views.angle = self.angle
        print(self.name, self.world_matrix[0, 2], self.world_matrix[1, 2])

    def draw(self, **kwargs):
        if isinstance(self.views, arcade.SpriteList):
            self.sync()
            self.views.draw(**kwargs)
        elif isinstance(self.views, GridMesh):
            self.sync()
            self.views.draw()

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
