import math
import time
from Const import GLOBAL_SCALE
from functools import partial


def is_blinking(bl, br):
    return bl < 0.25 or br < 0.25


def block_blinking(blink):
    return blink < EAR_MIN


def get_blink(dir, data):
    blinkL, blinkR = data["Blinking"]
    if dir == "L":
        return blinkL
    else:
        return blinkR


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


def face_response(live, data):
    live.angle = data["Roll"]
    return


last_valid_eye_x = 0
last_valid_eye_y = 0


def pupils_response(dir, live, data):
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
    live.add_x = x
    live.add_y = y
    if block_blinking(get_blink(dir, data)):
        live.sx = 0
    else:
        live.sx = 1
    return


pupils_response_l = partial(pupils_response, "L")
pupils_response_r = partial(pupils_response, "R")


def mouth_response(live, data):
    openness = data["MouthOpenness"]
    # Simple threshold logic (Logic Step 2)
    if openness < 0.1:
        new_state = "MouthClose"
    elif openness < 0.5:
        new_state = "MouthHalf"
    else:
        new_state = "MouthOpen"

    # Only swap texture if the state actually changed (Optimization)
    views = live.views
    if new_state != views.current_state:
        views.texture = views.state_textures[new_state]
        views.current_state = new_state
    return


from Const import *
from ValueUtils import map_range


def lid_response(dir, live, data):
    blinkL, blinkR = data["Blinking"]
    if dir == "L":
        blink = blinkL
    else:
        blink = blinkR
    # blink = self.filter_blink(current_time, blinkL)
    # update local
    target_y = -map_range(blink, EAR_MIN, EAR_MAX, EYE_CLOSED_Y, EYE_OPEN_Y)
    # self.local_x = self.base_local_x
    live.add_y = target_y
    close_progress = map_range(blink, EAR_MIN, EAR_MAX, 1.0, 0.0)
    # live.sy = 1.0 - (close_progress * 0.4)
    data[f"EyeLashClose{dir}"] = target_y
    data[f"EyeLashScale{dir}"] = 1.0 - (close_progress * 0.4)
    return


lid_response_l = partial(lid_response, "L")
lid_response_r = partial(lid_response, "R")


def lash_response(dir, live, data):
    live.add_y = data[f"EyeLashClose{dir}"]
    if block_blinking(get_blink(dir, data)):
        live.sy = -data[f"EyeLashScale{dir}"]
    else:
        live.sy = data[f"EyeLashScale{dir}"]


lash_response_l = partial(lash_response, "L")
lash_response_r = partial(lash_response, "R")


def white_response(dir, live, data):
    if block_blinking(get_blink(dir, data)):
        live.sy = 0
    else:
        live.sy = 1


white_response_l = partial(white_response, "L")
white_response_r = partial(white_response, "R")
