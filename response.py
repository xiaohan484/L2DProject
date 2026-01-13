import math
import time
from Const import GLOBAL_SCALE


def is_blinking(bl, br):
    return bl < 0.25 or br < 0.25


def body_response(live, data):
    # 呼吸頻率 (速度)
    BREATH_SPEED = 1.5
    # 呼吸幅度 (縮放比例，不用太大，0.01 代表 1%)
    BREATH_AMOUNT = 0.0025

    # 利用時間算出一個 -1 ~ 1 的波形
    breath_wave = math.sin(time.time() * BREATH_SPEED)
    # 應用在身體 (Root) 的 Y 軸縮放
    # 1.0 是原始大小，加上波形變化
    live.sy = GLOBAL_SCALE + breath_wave * BREATH_AMOUNT
    live.sx = GLOBAL_SCALE + breath_wave * (
        BREATH_AMOUNT * 0.1
    )  # X軸稍微跟著動一點點會更自然
    return


last_valid_eye_x = 0
last_valid_eye_y = 0


def pupils_response(live, data):
    bl, br = data["Blinking"]
    calibration = (0, 0)

    def convertPupils(target, calibration, blinking):
        global last_valid_eye_x, last_valid_eye_y
        MAX_W = 10  # 橢圓的長軸 (左右極限)
        MAX_H = 7  # 橢圓的短軸 (上下極限)
        dx, dy = target
        calibration_x, calibration_y = calibration
        dx -= calibration_x
        dy -= calibration_y
        if blinking:
            dx = last_valid_eye_x
            dy = last_valid_eye_y
        else:
            last_valid_eye_x = dx
            last_valid_eye_y = dy

        raw_x = dx * MAX_W
        raw_y = dy * MAX_H
        return raw_x, raw_y

    x, y = convertPupils(data["PupilsPos"], calibration, is_blinking(bl, br))
    add_x, add_y = data["FaceBaseOffset"]
    live.add_x = x + add_x
    live.add_y = y + add_y
    return


def mouth_response(live, data):
    openness = data["MouthOpenness"]
    # Simple threshold logic (Logic Step 2)
    if openness < 0.1:
        new_state = "MouthClose"
    elif openness < 0.3:
        new_state = "MouthOpen"
    else:
        new_state = "MouthHalf"

    # Only swap texture if the state actually changed (Optimization)
    views = live.views
    if new_state != views.current_state:
        views.texture = views.state_textures[new_state]
        views.current_state = new_state
    return
