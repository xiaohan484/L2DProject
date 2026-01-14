import json
import math
import os
import threading
from Const import *
from VTSprite import *
import arcade
from tracker import AsyncFaceTracker, FakeTracker
from ValueUtils import *
from mesh_renderer import GridMesh, SkinnedMesh, lerp_apply
from bone_system import load_bones
import time
import numpy as np
from live2d import Live2DPart
from response import *

mode_mesh = 0
mode_sprite = 1


front_shadow_z = 2
face_feature_z = 1
face_base_z = 0
back_hair_z = -1

func_table = {
    "Body": (0, body_response),
    "BackHair": (back_hair_z, None),
    "Face": (face_base_z, None),
    "FaceLandmark": (face_feature_z, None),
    "EyeLidL": (face_feature_z, lid_response_l),
    "EyeLidR": (face_feature_z, lid_response_r),
    "EyeWhiteL": (face_feature_z, white_response_l),
    "EyeWhiteR": (face_feature_z, white_response_r),
    "EyePupilL": (face_feature_z, pupils_response_l),
    "EyePupilR": (face_feature_z, pupils_response_r),
    "EyeLashL": (face_feature_z, lash_response_l),
    "EyeLashR": (face_feature_z, lash_response_r),
    "EyeBrowL": (face_feature_z, None),
    "EyeBrowR": (face_feature_z, None),
    "Mouth": (face_feature_z, mouth_response),
    "FrontHairLeft": (face_feature_z, None),
    "FrontHairMiddle": (face_feature_z, None),
    "FrontHairLeftShadow": (front_shadow_z, None),
    "FrontHairMiddleShadow": (front_shadow_z, None),
}


def create_live2dpart(ctx, name, mode, parent=None):
    if name in func_table:
        z, res = func_table[name]
    else:
        res = None
        z = None
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
        self.setup_scene()

    def notify_face_info(self, face_info):
        with self.lock:
            self.face_info = face_info

    def __del__(self):
        self.tracker.release()

    def create_live2dpart(self, name, parent=None):
        return create_live2dpart(self.ctx, name, mode_mesh, parent)

    def setup_scene(self):
        # --- 1. 建立根節點 (Face) ---
        # 假設你的 JSON 裡臉部叫做 "face" (如果不一樣請修改 key)
        # body
        body = self.create_live2dpart("Body")
        back_hair = self.create_live2dpart("BackHair", body)
        face = self.create_live2dpart("Face", body)
        landmarks = self.create_live2dpart("FaceLandmark", face)
        eye_lid_L = self.create_live2dpart("EyeLidL", face)
        eye_lid_R = self.create_live2dpart("EyeLidR", face)
        eye_white_L = self.create_live2dpart("EyeWhiteL", face)
        eye_white_R = self.create_live2dpart("EyeWhiteR", face)
        eye_pupil_L = self.create_live2dpart("EyePupilL", face)
        eye_pupil_R = self.create_live2dpart("EyePupilR", face)
        eye_lash_L = self.create_live2dpart("EyeLashL", face)
        eye_lash_R = self.create_live2dpart("EyeLashR", face)
        eye_brow_L = self.create_live2dpart("EyeBrowL", face)
        eye_brow_R = self.create_live2dpart("EyeBrowR", face)
        mouth = self.create_live2dpart("Mouth", face)
        front_hair_left = self.create_live2dpart("FrontHairLeft", face)
        front_hair_middle = self.create_live2dpart("FrontHairMiddle", face)
        front_hair_middle_shadow = self.create_live2dpart("FrontHairShadowMiddle", face)
        front_hair_left_shadow = self.create_live2dpart("FrontHairShadowLeft", face)

        self.drawlist = [
            back_hair,
            body,
            face,
            landmarks,
            eye_white_L,
            eye_white_R,
            eye_pupil_L,
            eye_pupil_R,
            eye_lid_L,
            eye_lid_R,
            eye_lash_L,
            eye_lash_R,
            eye_brow_L,
            eye_brow_R,
            mouth,
            front_hair_left_shadow,
            front_hair_middle_shadow,
            front_hair_left,
            front_hair_middle,
        ]
        self.body = body

        # self.eye_lid_L.set_depend(
        #    children={
        #        "eye_lash": self.eye_lash_L,
        #        "eye_white": self.eye_white_L,
        #        "eye_pupil": self.eye_pupil_L,
        #    }
        # )
        # self.eye_lid_R.set_depend(
        #    children={
        #        "eye_lash": self.eye_lash_R,
        #        "eye_white": self.eye_white_R,
        #        "eye_pupil": self.eye_pupil_R,
        #    }
        # )
        # self.groups = {
        #    "FaceFeature": [
        #        # self.face_landmarks,
        #        self.mouth,
        #        self.eye_white_L,
        #        self.eye_white_R,
        #        self.eye_lid_L,
        #        self.eye_lid_R,
        #        self.eye_lash_L,
        #        self.eye_lash_R,
        #        self.eye_brow_L,
        #        self.eye_brow_R,
        #        self.hair_left_mesh,
        #        self.hair_middle_mesh,
        #        self.hair_right_mesh,
        #    ],
        #    "FrontShadow": [
        #        self.hair_front_left_shadow_mesh,
        #        self.hair_front_right_shadow_mesh,
        #        self.hair_front_middle_shadow_mesh,
        #    ],
        #    "FaceBase": [self.face],
        #    "BackHair": [self.back_hair_mesh],
        # }

    def on_draw(self):
        self.clear()
        self.camera.use()

        for obj in self.drawlist:
            obj.draw()

    def update_pose(self, face_info):
        yaw, pitch, roll = filterHead(face_info["Pose"])
        # 2. 定義強度 (這就是你要一直調的參數)
        # 代表轉 1 度，像素要移動多少 px
        PARALLAX_X_STRENGTH = -0.4
        PARALLAX_Y_STRENGTH = 0.5
        # 3. 計算各層的位移量 (Offset)
        # Layer Depth Multipliers (深度乘數)
        # 前面動得快，後面動得慢
        OFFSET_FRONT = 1.0001  # 前髮、五官
        OFFSET_MID = 1.0  # 臉型
        OFFSET_BACK = 0.9999  # 後髮
        OFFSET_FRONT_SHADOW = 0.5
        # 計算 X 軸位移
        move_x_front_shadow = yaw * PARALLAX_X_STRENGTH * OFFSET_FRONT_SHADOW
        move_x_front = yaw * PARALLAX_X_STRENGTH * OFFSET_FRONT
        move_x_mid = yaw * PARALLAX_X_STRENGTH * OFFSET_MID
        move_x_back = yaw * PARALLAX_X_STRENGTH * OFFSET_BACK
        # 計算 Y 軸位移
        move_y_front_shadow = pitch * PARALLAX_Y_STRENGTH * OFFSET_FRONT_SHADOW
        move_y_front = pitch * PARALLAX_Y_STRENGTH * OFFSET_FRONT
        move_y_mid = pitch * PARALLAX_Y_STRENGTH * OFFSET_MID
        move_y_back = pitch * PARALLAX_Y_STRENGTH * OFFSET_BACK

        # 4. 應用到 Sprites (記得加上原本的基礎位置)
        # 假設 base_x 是螢幕中心
        face_info["FrontShadowOffset"] = (move_x_front_shadow, 4 * move_y_front_shadow)
        face_info["FaceFeatureOffset"] = (move_x_front, move_y_front)
        face_info["FaceBaseOffset"] = (move_x_mid, move_y_mid)
        face_info["BackHairOffset"] = (move_x_back, move_y_back)

        # self.face_landmarks.update(move_x_front, move_y_front)

        # for group, sprites in self.groups.items():
        #    add_x, add_y = face_info[group + "Offset"]
        #    for s in sprites:
        #        s.local_x = s.base_local_x + add_x
        #        s.local_y = s.base_local_y + add_y
        # self.body.local_x = self.body.base_local_x + move_x_back
        # self.body.local_y = self.body.base_local_y - move_y_back

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
        self.body.update(face_info)


if __name__ == "__main__":
    # 載入設定檔
    # game = MyGame(tracker=FakeTracker())
    game = MyGame(tracker=AsyncFaceTracker())
    arcade.run()
