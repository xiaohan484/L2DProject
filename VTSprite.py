import arcade
from Const import *
import math

class VTSprite(arcade.Sprite):
    """
    支援父子階層與錨點的 Sprite
    """
    def __init__(self, filename, scale=1.0, parent=None, data_key=None):
        super().__init__(filename, scale)
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
        
        # 錨點 (0.0 ~ 1.0)，預設中心
        self.anchor_x_ratio = 0.5
        self.anchor_y_ratio = 0.5

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
            
            # TODO: 這裡尚未實作 "自身錨點旋轉" (Self-Pivot)，
            # 目前旋轉是以 Sprite 中心為主。如果要讓頭部繞著脖子轉，
            # 需要再加一段 offset math，但 MVP 先這樣即可。

        # 遞迴更新孩子
        for child in self.children:
            child.update_transform()

from filters import OneEuroFilter
filter_eye_x = OneEuroFilter(min_cutoff=0.5, beta=0.5)
filter_eye_y = OneEuroFilter(min_cutoff=0.5, beta=0.5)
# Blink Linking 狀態變數
last_valid_eye_x = 0.0
last_valid_eye_y = 0.0
def convertPupils(target, calibration, blinking):
    MAX_W = 10  # 橢圓的長軸 (左右極限)
    MAX_H = 7   # 橢圓的短軸 (上下極限)
    dx ,dy = target
    calibration_x,calibration_y = calibration
    dx-= calibration_x
    dy-= calibration_y
    global filter_eye_x,filter_eye_y,last_valid_eye_x,last_valid_eye_y
    if blinking:
        dx = last_valid_eye_x
        dy = last_valid_eye_y
    else:
        import time
        current_time = time.time()
        dx = filter_eye_x(current_time, dx)
        dy = filter_eye_y(current_time, dy)
        last_valid_eye_x = dx
        last_valid_eye_y = dy

    raw_x = dx * MAX_W 
    raw_y = dy * MAX_H 
    norm_dist = (raw_x / MAX_W)**2 + (raw_y / MAX_H)**2
    if norm_dist > 1:
        scale = 1 / math.sqrt(norm_dist)
        final_x = raw_x * scale
        final_y = raw_y * scale
    else:
        final_x = raw_x
        final_y = raw_y
    return final_x,final_y

filter_blink_l = OneEuroFilter(min_cutoff=0.1, beta=50.0)
filter_blink_r = OneEuroFilter(min_cutoff=0.1, beta=100.0)
def filterBlink(blink):
    import time
    current_time = time.time()
    blinkL, blinkR = blink
    blinkL = filter_blink_l(current_time, blinkL)
    blinkR = filter_blink_r(current_time, blinkR)
    return blinkL,blinkR

# 針對頭部動作，通常 beta (速度響應) 可以設小一點，讓頭部感覺比較重、比較穩
head_yaw_filter = OneEuroFilter(min_cutoff=0.01, beta=0.1)
head_pitch_filter = OneEuroFilter(min_cutoff=0.01, beta=0.1)
head_roll_filter = OneEuroFilter(min_cutoff=0.01, beta=0.1)
def filterHead(head_pose):
    import time
    current_time = time.time()
    yaw,pitch,roll = head_pose
    print(yaw,pitch,roll)
    yaw = head_yaw_filter(current_time, yaw)
    pitch = head_pitch_filter(current_time, pitch)
    roll = head_roll_filter(current_time, roll)
    return yaw,pitch,roll