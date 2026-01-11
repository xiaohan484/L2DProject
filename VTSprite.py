import math
import time
import arcade
from pubsub import pub
from Const import *
from filters import OneEuroFilter
from ValueUtils import map_range


def is_blinking(bl, br):
    return bl < 0.25 or br < 0.25


class VTSprite(arcade.Sprite):
    """
    支援父子階層與錨點的 Sprite
    """

    def __init__(self, filename, scale=1.0, parent=None, data_key=None):
        super().__init__(filename, scale)
        print(filename)
        self.parent = parent
        self.children = []
        if parent:
            parent.children.append(self)

        # 儲存從 JSON 讀來的原始設定 (方便 Debug)
        self.data_key = data_key

        self.base_local_x = 0
        self.base_local_y = 0

        # 這些是 "相對" 於父物件的屬性 (Local Transform)
        self.local_scale_x = 1.0
        self.local_scale_y = 1.0
        self.local_x = 0
        self.local_y = 0
        self.local_angle = 0
        self.local_scale_y = 1.0

    def update_transform(self):
        """核心：遞迴更新座標"""
        if self.parent:
            # 1. 取得父物件資訊
            p_x, p_y = self.parent.center_x, self.parent.center_y
            p_angle = self.parent.angle
            p_scale = self.parent.scale

            # 2. 計算旋轉 (父角度 + 本地角度)
            # 這裡的數學確保了 "跟著爸爸轉"
            rad = math.radians(p_angle)

            # 旋轉公式
            rot_x = self.local_x * math.cos(rad) - self.local_y * math.sin(rad)
            rot_y = -(self.local_x * math.sin(rad) + self.local_y * math.cos(rad))

            # 3. 更新自己的全域座標
            self.center_x = p_x + rot_x * p_scale[0]
            self.center_y = p_y + rot_y * p_scale[1]
            self.angle = p_angle + self.local_angle
            self.scale = p_scale

            # 特別處理：眨眼縮放 (Y軸)
            self.height = self.texture.height * self.scale[1] * self.local_scale_y
        else:
            self.center_x = self.local_x
            self.center_y = self.local_y

        # 遞迴更新孩子
        for child in self.children:
            child.update_transform()


class MouthSprite(VTSprite):
    def __init__(
        self, closed_path, half_path, open_path, scale=1.0, parent=None, data_key=None
    ):
        # Initialize the parent Sprite with the default (closed) image
        super().__init__(
            filename=closed_path, scale=scale, parent=parent, data_key=data_key
        )

        # Pre-load all textures into a dictionary for fast access
        # We use strict naming to manage states
        self.state_textures = {
            "closed": arcade.load_texture(closed_path),
            "half": arcade.load_texture(half_path),
            "open": arcade.load_texture(open_path),
        }

        self.current_state = "closed"

    def update_state(self, face_info):
        """
        Updates the sprite texture based on the openness value (0.0 to 1.0).
        """
        openness = face_info["MouthOpenness"]
        # Simple threshold logic (Logic Step 2)
        if openness < 0.1:
            new_state = "closed"
        elif openness < 0.3:
            new_state = "half"
        else:
            new_state = "open"

        # Only swap texture if the state actually changed (Optimization)
        if new_state != self.current_state:
            self.texture = self.state_textures[new_state]
            self.current_state = new_state


class PupilsSprite(VTSprite):
    def __init__(self, filename, scale=1.0, parent=None, data_key=None):
        super().__init__(filename, scale, parent=parent, data_key=data_key)
        self.filter_eye_x = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        self.filter_eye_y = OneEuroFilter(min_cutoff=0.5, beta=0.5)
        self.last_valid_eye_x = 0.0
        self.last_valid_eye_y = 0.0
        # self.calibration = (-0.5867841357630763, -0.5041574138173885)
        self.calibration = (0, 0)
        self.x = 0
        self.y = 0

    def update_state(self, face_info):
        bl, br = face_info["Blinking"]
        x, y = self.convertPupils(
            face_info["PupilsPos"], self.calibration, is_blinking(bl, br)
        )
        add_x, add_y = face_info["FaceBaseOffset"]
        x += add_x
        y += add_y
        self.local_x = self.base_local_x + x
        self.local_y = self.base_local_y + y

    def convertPupils(self, target, calibration, blinking):
        MAX_W = 10  # 橢圓的長軸 (左右極限)
        MAX_H = 7  # 橢圓的短軸 (上下極限)
        dx, dy = target
        calibration_x, calibration_y = calibration
        dx -= calibration_x
        dy -= calibration_y
        if blinking:
            dx = self.last_valid_eye_x
            dy = self.last_valid_eye_y
        else:
            self.last_valid_eye_x = dx
            self.last_valid_eye_y = dy

        raw_x = dx * MAX_W
        raw_y = dy * MAX_H
        return raw_x, raw_y


class LidSprite(VTSprite):
    def __init__(self, filename, scale=1.0, parent=None, data_key=None):
        super().__init__(filename, scale, parent=parent, data_key=data_key)
        self.x = 0
        self.y = 0
        self.info = None
        if "LidL" in filename:
            self.dir = "L"
            self.filter_blink = OneEuroFilter(min_cutoff=0.1, beta=50.0)
        else:
            self.dir = "R"
            self.filter_blink = OneEuroFilter(min_cutoff=0.1, beta=100.0)
        self.depend = {}

    def set_depend(self, children):
        self.depend = children

    def update_state(self, face_info):
        blinkL, blinkR = face_info["Blinking"]
        if self.dir == "L":
            blink = blinkL
        else:
            blink = blinkR
        current_time = time.time()
        # blink = self.filter_blink(current_time, blinkL)

        # update local
        target_y = map_range(blink, EAR_MIN, EAR_MAX, EYE_CLOSED_Y, EYE_OPEN_Y)
        # self.local_x = self.base_local_x
        self.local_y += target_y
        close_progress = map_range(blink, EAR_MIN, EAR_MAX, 1.0, 0.0)
        target_scale_y = 1.0 - (close_progress * 0.4)
        self.depend["eye_lash"].local_y += int(target_y)

        if blink < EAR_MIN:
            self.depend["eye_lash"].local_scale_y = -1 * target_scale_y
            self.depend["eye_white"].local_scale_y = 0
            self.depend["eye_pupil"].local_scale_y = 0
        else:
            self.depend["eye_lash"].local_scale_y = 1 * target_scale_y
            self.depend["eye_white"].local_scale_y = 1
            self.depend["eye_pupil"].local_scale_y = 1
        return


# 針對頭部動作，通常 beta (速度響應) 可以設小一點，讓頭部感覺比較重、比較穩
head_yaw_filter = OneEuroFilter(min_cutoff=10, beta=0.1)
head_pitch_filter = OneEuroFilter(min_cutoff=10, beta=0.1)
head_roll_filter = OneEuroFilter(min_cutoff=10, beta=0.1)


def filterHead(head_pose):

    current_time = time.time()
    yaw, pitch, roll = head_pose
    yaw = head_yaw_filter(current_time, yaw)
    pitch = head_pitch_filter(current_time, pitch)
    roll = head_roll_filter(current_time, roll)
    return yaw, pitch, roll


if __name__ == "__main__":
    prefix = "assets/sample_model/processed/"
    mouth = MouthSprite(
        closed_path=prefix + "MouthClose.png",
        half_path=prefix + "MouthHalf.png",
        open_path=prefix + "MouthOpen.png",
    )
    print(mouth.base_local_x)
