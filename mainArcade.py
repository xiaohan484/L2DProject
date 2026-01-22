import os
import threading
from Const import *
from VTSprite import *
import arcade
from tracker import AsyncFaceTracker, FakeTracker
from ValueUtils import *
from mesh_renderer import GridMesh
from bone_system import load_bones
import numpy as np
from live2d import (
    Live2DPart,
    back_hair_z,
    face_base_z,
    face_feature_z,
    front_shadow_z,
    body_base_z,
)
from response import *
from collections import OrderedDict

mode_mesh = 0
mode_sprite = 1


func_table = OrderedDict(
    {
        "Body": (body_base_z, body_response, None),
        "BackHair": (back_hair_z, None, "Body"),
        "Face": (face_base_z, None, "Body"),
        "EyeWhiteL": (face_feature_z, white_response_l, "Face"),
        "EyeWhiteR": (face_feature_z, white_response_r, "Face"),
        "EyePupilL": (face_feature_z, pupils_response_l, "Face"),
        "EyePupilR": (face_feature_z, pupils_response_r, "Face"),
        "FaceLandmark": (face_feature_z, None, "Face"),
        "EyeLidL": (face_feature_z, lid_response_l, "Face"),
        "EyeLidR": (face_feature_z, lid_response_r, "Face"),
        "EyeLashL": (face_feature_z, lash_response_l, "Face"),
        "EyeLashR": (face_feature_z, lash_response_r, "Face"),
        "EyeBrowL": (face_feature_z, None, "Face"),
        "EyeBrowR": (face_feature_z, None, "Face"),
        "Mouth": (face_feature_z, mouth_response, "Face"),
        "FrontHairShadowLeft": (front_shadow_z, None, "Face"),
        "FrontHairShadowMiddle": (front_shadow_z, None, "Face"),
        "FrontHairLeft": (face_feature_z, None, "Face"),
        "FrontHairMiddle": (face_feature_z, None, "Face"),
    }
)


def create_live2dpart(ctx):
    lives = OrderedDict({})
    for name, (_, _, parent) in func_table.items():
        if parent is not None:
            parent = lives[parent]
        lives[name] = create_live2dpart_each(ctx, name, parent)
    root = lives["Body"]
    lives.move_to_end("BackHair", last=False)
    return lives, root


def create_live2dpart_each(ctx, name, parent):
    z, res, _ = func_table[name]
    data = MODEL_DATA[name]
    textures = {}
    for path in data["filename"]:
        filepath = os.path.join("assets/sample_model/processed", path)
        p = path.removesuffix(".png")
        textures[p] = filepath

    raw_global_x = data["global_center_x"]
    raw_global_y = data["global_center_y"]
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
        )


def load_local_position(global_pos, parent_global_pos, global_scale):
    # Calculate the relative distance in global space
    delta_x = global_pos[0] - parent_global_pos[0]
    delta_y = global_pos[1] - parent_global_pos[1]

    # Scale down the delta to get the original local position
    local_x = delta_x / global_scale
    local_y = delta_y / global_scale

    return local_x, local_y


class MyGame(arcade.Window):
    def __init__(self, tracker):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.tracker = tracker
        self.calibration = (-0.5867841357630763, -0.5041574138173885)
        arcade.set_background_color(arcade.color.GREEN)
        self.all_sprites = arcade.SpriteList()
        self.face_info = None
        self.lock = threading.Lock()
        pub.subscribe(self.notify_face_info, "FaceInfo")

        self.hair_physics = SpringDamper(stiffness=0.01, damping=0.6)
        self.hair_physics_pendulum = PendulumPhysics(
            stiffness=0.01, damping=0.6, mass=1.0, gravity_power=1.0
        )
        self.last_yaw = 0
        self.total_time = 0
        self.camera = arcade.Camera2D()
        self.camera.position = (0, 950)
        self.camera.zoom = 1.5
        self.hair_bones = None
        self.lives, self.root = create_live2dpart(self.ctx)

    def notify_face_info(self, face_info):
        with self.lock:
            self.face_info = face_info

    def __del__(self):
        self.tracker.release()

    def on_draw(self):
        self.clear()
        self.camera.use()

        for _, obj in self.lives.items():
            obj.draw()

    def update_pose(self, face_info):
        yaw, pitch, roll = filterHead(face_info["Pose"])
        face_info["Yaw"] = yaw
        face_info["Pitch"] = pitch
        face_info["Roll"] = roll
        ## front hair physics
        # yaw_velocity = yaw - self.last_yaw

        ## RD 技巧：限制最大速度 (Clamp)，避免追蹤丟失瞬間頭部瞬移導致頭髮爆炸
        # yaw_velocity = max(-5.0, min(yaw_velocity, 5.0))

        # self.last_yaw = yaw
        ## 3. 更新物理系統
        ## 這裡的 scaling_factor 很重要，用來把「角度差」轉換成「彎曲力道」
        ## 負號是因為慣性方向相反 (頭向左，髮尾甩向右)
        # force = -yaw_velocity * 0.01
        # bend_val, _ = self.hair_physics.update(force)
        ## 4. 驅動網格
        # for front_hair in [
        #    self.hair_middle_mesh,
        #    self.hair_right_mesh,
        #    self.hair_front_middle_shadow_mesh,
        #    self.hair_front_right_shadow_mesh,
        # ]:
        #    front_hair.apply_bend(bend_val, self.total_time)
        # self.back_hair_mesh.apply_bend(-bend_val * 0.2, self.total_time)

        ## pendulum
        # head_angle = self.hair_bones[1].angle
        # gravity_target = head_angle * self.hair_physics_pendulum.gravity_power
        # final_angle = self.hair_physics_pendulum.update(
        #    target_angle_offset=gravity_target, input_force=force
        # )
        # gain = 1.1
        # for i, b in enumerate(self.hair_bones):
        #    if i == 0:
        #        b.angle = 0
        #    elif i == 1:
        #        b.angle = head_angle + (-5 * final_angle) * 0.1
        #    else:
        #        b.angle = -5 * final_angle * gain
        #        gain *= 1.1
        # self.hair_bones[0].update()
        # self.hair_left_mesh.update_skinning()
        # self.hair_front_left_shadow_mesh.update_skinning()

    def on_update(self, delta_time):
        with self.lock:
            face_info = self.face_info
        if face_info is None:
            return
        self.total_time += delta_time
        self.update_pose(face_info)
        self.root.update(face_info)


if __name__ == "__main__":
    # 載入設定檔
    # game = MyGame(tracker=FakeTracker())
    game = MyGame(tracker=AsyncFaceTracker())
    arcade.run()
