import sys
import os
import arcade
import json

# 設定路徑以便引用 engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bone_system import Bone
from mesh_renderer import SkinnedMesh

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


# ... (前面的 import 不變) ...

# 定義編輯模式
MODE_BONE = 0  # 架骨架模式
MODE_WEIGHT = 1  # 刷權重模式


class RiggingEditor(arcade.Window):
    def __init__(self):
        super().__init__(
            SCREEN_WIDTH, SCREEN_HEIGHT, "Level 7: Rigging & Skinning Editor"
        )
        arcade.set_background_color(arcade.color.GREEN)

        self.mode = MODE_BONE  # 預設先架骨頭

        # 1. 骨骼資料
        self.bones = []
        self.selected_bone = None  # 目前選中的骨頭物件

        # 嘗試載入骨骼，如果沒有則建立預設的
        if os.path.exists("../assets/hair_bones.json"):
            self.load_bones("../assets/hair_bones.json")
        else:
            # 預設至少要有一根 Root
            root = Bone("Root", 0, 0)
            self.bones.append(root)
            self.selected_bone = root

        from Const import MODEL_PATH

        # 2. 載入 Mesh (注意：Mesh 初始化時需要骨頭列表)
        self.mesh = SkinnedMesh(self.ctx, f"{MODEL_PATH}/HairFront.png", self.bones)
        self.mesh.center_x = SCREEN_WIDTH / 2
        self.mesh.center_y = SCREEN_HEIGHT / 2
        self.mouse_x = 0
        self.mouse_y = 0

        # 嘗試載入權重
        if os.path.exists("assets/hair_weights.json"):
            self.mesh.load_weights_from_file("assets/hair_weights.json")

    def load_bones(self, filepath):
        """從 JSON 讀取骨骼並重建父子關係"""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.bones = []
        bone_map = {}  # name -> Bone Object

        # 第一輪：建立物件
        for b_data in data:
            bone = Bone(b_data["name"], b_data["x"], b_data["y"])
            self.bones.append(bone)
            bone_map[bone.name] = bone

        # 第二輪：建立連結
        for b_data in data:
            if b_data["parent"]:
                child = bone_map[b_data["name"]]
                parent = bone_map[b_data["parent"]]
                child.parent = parent
                parent.children.append(child)

        # 預設選中第一根
        if self.bones:
            self.selected_bone = self.bones[0]
        print(f"已載入 {len(self.bones)} 根骨頭。")

    def save_bones(self, filepath):
        """將骨骼存檔"""
        data = []
        for b in self.bones:
            parent_name = b.parent.name if b.parent else None
            data.append({"name": b.name, "x": b.x, "y": b.y, "parent": parent_name})
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print("骨骼已存檔！")

    def on_draw(self):
        self.clear()

        # 畫 Mesh
        self.mesh.draw()

        # 畫骨頭
        for i, bone in enumerate(self.bones):
            bx, by = bone.get_world_position()

            # 顏色邏輯：
            # 骨頭模式下：選中=黃色，其他=白色
            # 權重模式下：選中=該權重的代表色(例如綠色)，其他=灰色
            color = arcade.color.WHITE
            if self.mode == MODE_BONE:
                color = (
                    arcade.color.YELLOW
                    if bone == self.selected_bone
                    else arcade.color.BLUE_GRAY
                )
            else:
                color = (
                    arcade.color.GREEN
                    if bone == self.selected_bone
                    else arcade.color.GRAY
                )

            arcade.draw_circle_filled(bx, by, 8, color)
            if bone.parent:
                px, py = bone.parent.get_world_position()
                arcade.draw_line(bx, by, px, py, arcade.color.WHITE, 2)
            if i > 0:
                px, py = last_bone
                arcade.draw_line(bx, by, px, py, arcade.color.WHITE, 2)
            last_bone = bx, by

        # 畫頂點 (只在權重模式顯示，避免畫面太亂)
        if self.mode == MODE_WEIGHT:
            vertices = self.mesh.current_vertices.reshape(-1, 4)
            for i, v in enumerate(vertices):
                vx = v[0] + self.mesh.center_x
                vy = v[1] + self.mesh.center_y

                # 根據主要權重變色
                weights = self.mesh.vertex_weights[i]
                main_bone_idx = weights[0][0]

                # 如果這個點歸「目前選中的骨頭」管，亮綠色，否則紅色
                # (這裡簡化顯示邏輯)
                is_controlled_by_selected = (
                    self.bones[main_bone_idx] == self.selected_bone
                )
                color = (
                    arcade.color.GREEN
                    if is_controlled_by_selected
                    else arcade.color.RED
                )

                arcade.draw_point(vx, vy, color, 3)

        # UI 狀態顯示
        mode_str = (
            "BONE MODE (Add/Move)" if self.mode == MODE_BONE else "WEIGHT MODE (Paint)"
        )
        arcade.draw_text(
            f"Mode: {mode_str} [Press B / W]", 10, 580, arcade.color.WHITE, 14
        )

        sel_name = self.selected_bone.name if self.selected_bone else "None"
        arcade.draw_text(f"Selected: {sel_name}", 10, 560, arcade.color.WHITE, 14)

        arcade.draw_text(
            "S: Save All | N: New Bone (Child) | Click: Select/Move/Paint",
            10,
            10,
            arcade.color.WHITE,
            12,
        )

    def on_mouse_motion(self, x, y, dx, dy):
        """
        Called whenever the mouse moves.
        """
        self.mouse_x = x
        self.mouse_y = y
        return

    def on_key_press(self, key, modifiers):
        if key == arcade.key.B:
            self.mode = MODE_BONE
        elif key == arcade.key.W:
            self.mode = MODE_WEIGHT
        elif key == arcade.key.S:
            self.save_bones("../assets/hair_bones.json")
            self.mesh.save_weights_to_file("../assets/hair_weights.json")
        elif key == arcade.key.N and self.mode == MODE_BONE:
            # 【新增骨頭】在滑鼠位置附近，或是目前骨頭的下方
            if self.selected_bone:
                print(self.mouse_x, self.mouse_y)
                new_bone = Bone(
                    f"Bone_{len(self.bones)}",
                    self.mouse_x,
                    self.mouse_y,  # 預設往下長
                    parent=None,
                )
                self.bones.append(new_bone)
                # 記得更新 Mesh 裡的骨頭列表引用
                self.mesh.bones = self.bones
                self.selected_bone = new_bone
                print("新增骨頭！")

    def on_mouse_press(self, x, y, button, modifiers):
        # 通用邏輯：先檢查有沒有點到骨頭 (選取骨頭)
        min_dist = 15 * 15
        clicked_bone = None
        for bone in self.bones:
            bx, by = bone.get_world_position()
            if (bx - x) ** 2 + (by - y) ** 2 < min_dist:
                clicked_bone = bone
                break

        if clicked_bone:
            self.selected_bone = clicked_bone
            return  # 如果點到骨頭，就只做選取，不刷權重

        # 模式分歧
        if self.mode == MODE_BONE:
            # 骨頭模式：如果沒點到骨頭，且按著滑鼠，可能是要拖曳目前選中的骨頭
            # 這裡簡化：點擊空白處移動目前骨頭
            print("bone mode:", x, y)
            if self.selected_bone:
                self.selected_bone.x = x
                self.selected_bone.y = y
                self.selected_bone.update()  # 更新矩陣
                # 骨頭動了，Local Position 就失效了，理論上要重算，
                # 但 Rigging 階段通常假設 Mesh 跟著骨頭走，先不重算

        elif self.mode == MODE_WEIGHT:
            print("bone weight")
            # 權重模式：刷網格
            if self.selected_bone:
                # 找出選中骨頭的 index
                bone_idx = self.bones.index(self.selected_bone)

                vertices = self.mesh.current_vertices.reshape(-1, 4)
                paint_radius = 30 * 30

                changed = False
                for i, v in enumerate(vertices):
                    vx = v[0] + self.mesh.center_x
                    vy = v[1] + self.mesh.center_y
                    if (vx - x) ** 2 + (vy - y) ** 2 < paint_radius:
                        # 綁定給當前骨頭
                        self.mesh.vertex_weights[i] = [(bone_idx, 1.0)]
                        changed = True

                if changed:
                    self.mesh.recalculate_local_positions()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # 允許拖曳
        self.on_mouse_press(x, y, buttons, modifiers)


if __name__ == "__main__":
    window = RiggingEditor()
    arcade.run()
