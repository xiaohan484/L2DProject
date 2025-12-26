import pygame
import os
import math
from dataclasses import dataclass

CAM_X = 350
CAM_Y = 100
CAM_W = 600
CAM_H = 500
FPS = 30
L_BLINK_THRESHOLD = 0.25
R_BLINK_THRESHOLD = 0.2

# 用來做平滑移動 (避免抖動)
current_dx = 0.0
current_dy = 0.0

def load_img(name):
    path = os.path.join("assets", name)
    try:
        img = pygame.image.load(path).convert_alpha()
        return img
    except FileNotFoundError:
        print(f"找不到圖片: {name}，請檢查檔名")
        return pygame.Surface((1, 1)) # 回傳空圖片避免當機

def blitComponent(screen, comp, offset = (0,0)):
    offset_x,offset_y = offset
    screen.blit(comp, (offset_x - CAM_X, offset_y - CAM_Y))

def evaluatePupilOffset(dx,dy, isFake):
    # 設定你的極限範圍 (試著調整這兩個數字)
    MAX_W = 7  # 橢圓的長軸 (左右極限)
    MAX_H = 7   # 橢圓的短軸 (上下極限)
    if isFake:
        raw_x = (dx - CAM_W/2) * 0.1
        raw_y = (dy - CAM_H/2) * 0.1
    else:
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

def draw_eye_masked(screen, white_img, pupil_img, pupil_offset):
    """
    這是一個遮罩繪製函數：
    1. 建立一個暫存畫布 (Surface)
    2. 先畫眼珠
    3. 再把眼白疊上去，並使用 "乘法" 運算
       - 眼白透明的地方(Alpha=0) * 眼珠 = 透明 (眼珠消失)
       - 眼白實色的地方(Alpha=1) * 眼珠 = 眼珠 (眼珠顯示)
    """
    # 建立一個跟眼白一樣大的透明畫布
    temp_surf = pygame.Surface(white_img.get_size(), pygame.SRCALPHA)
    
    # 1. 在暫存畫布上畫眼珠 (記得座標要加上相對位置)
    # 這裡的座標是相對於 "眼白左上角" 的
    # 假設眼珠原本中心跟眼白中心是對齊的，我們只要加 offset
    pupil_x = (white_img.get_width() - pupil_img.get_width()) / 2 + pupil_offset[0]
    pupil_y = (white_img.get_height() - pupil_img.get_height()) / 2 + pupil_offset[1]
    
    temp_surf.blit(pupil_img, (pupil_x, pupil_y))
    
    # 2. 【魔法步驟】畫上眼白作為遮罩
    # 使用 BLEND_RGBA_MULT：保留兩者重疊的部分
    # 注意：這要求你的 eye_white.png 在看得到的地方要是 "純白" 且 "不透明"
    temp_surf.blit(white_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # 3. 把處理好的整顆眼睛畫回主螢幕
    screen_x = 0 - CAM_X # 這裡需要你眼白的原始全域座標，假設是 (0,0)
    screen_y = 0 - CAM_Y
    screen.blit(temp_surf, (screen_x, screen_y))


@dataclass
class TrackingResult:
    pupilsX : float = 0
    pupilsY : float = 0
    blinkL: float = 0
    blinkR: float = 0

def map_range(value, in_min, in_max, out_min, out_max):
    """
    將數值從一個區間映射到另一個區間，並限制範圍 (Clamp)
    例如: 把 0.25(閉) ~ 0.5(開) 映射成 0.0 ~ 1.0
    """
    # 1. 先計算原始比例
    slope = (out_max - out_min) / (in_max - in_min)
    output = out_min + slope * (value - in_min)
    
    # 2. 限制範圍 (Clamp)，確保不會出現負數或超過 1.0
    return max(out_min, min(out_max, output))

class Face2DDraw:
    def __init__(self, isFake: bool, screen):
        self.isFake = isFake
        self.screen = screen
        self.calibration = (0,0)
        self.debug_font = pygame.font.SysFont("Arial", 24) # 或 "Consolas"
        self.components = {
            "img_face"        : load_img("Face.png"),
            "img_eye_white_L" : load_img("EyeWhiteL.png"),
            "img_eye_white_R" : load_img("EyeWhiteR.png"),
            "img_eye_pupil_L" : load_img("EyePupilL.png"),
            "img_eye_pupil_R" : load_img("EyePupilR.png"),
            "img_eye_lash_L"  : load_img("EyeLashL.png"),
            "img_eye_lash_R"  : load_img("EyeLashR.png"),
            "img_eye_lash_close_L"  : load_img("EyeLashLClose.png"),
            "img_eye_lash_close_R"  : load_img("EyeLashRClose.png"),
            "img_eyebrow_L"   : load_img("EyeBrowL.png"),
            "img_eyebrow_R"   : load_img("EyeBrowR.png"),
            "img_hair_front"  : load_img("HairFront.png")
        }
    def setCalib(self,calibration):
        self.calibration = calibration
    def extractPupilsOffSet(self, result:TrackingResult):
        calibration_x,calibration_y = self.calibration
        target_dx, target_dy = result.pupilsX,result.pupilsY
        target_dx -= calibration_x
        target_dy -= calibration_y
        debug_text = f"X: {target_dx:.2f}| {calibration_x:.2f}  Y: {target_dy:.2f}"
        color = (255, 0, 0) if (target_dx == 0 and target_dy == 0) else (0, 0, 255)
        text_surf = self.debug_font.render(debug_text, True, color)
        help_text = self.debug_font.render("Press 'C' to Center Eyes", True, (50, 50, 50))
        self.screen.blit(text_surf, (10, 10)) # 畫在左上角
        self.screen.blit(help_text, (10, 40))
        return evaluatePupilOffset(target_dx, target_dy,self.isFake)

    def drawEyePupils(self, result:TrackingResult, mode):
        offset = self.extractPupilsOffSet(result)

        comps = self.components
        if "L" in mode:
            draw_eye_masked(self.screen, comps["img_eye_white_L"], \
                            comps["img_eye_pupil_L"], offset)
        if "R" in mode:
            draw_eye_masked(self.screen, comps["img_eye_white_R"], \
                            comps["img_eye_pupil_R"], offset)

    def drawEye(self, result: TrackingResult):
        comps = self.components
        screen = self.screen

        if  result.blinkL < L_BLINK_THRESHOLD:
            blitComponent(screen, comps["img_eye_lash_close_L"])
        else:
            blitComponent(screen, comps["img_eye_white_L"])
            blitComponent(screen, comps["img_eye_lash_L"])
            self.drawEyePupils(result, "L")

        if result.blinkR < R_BLINK_THRESHOLD:
            blitComponent(screen, comps["img_eye_lash_close_R"])
        else:
            blitComponent(screen, comps["img_eye_white_R"])
            blitComponent(screen, comps["img_eye_lash_R"])
            self.drawEyePupils(result, "R")
        print(f"{result.blinkL:.3f} | {result.blinkR:.3f}")
    def draw(self, result: TrackingResult):
        comps = self.components
        screen = self.screen
        blitComponent(screen, comps["img_face"])
        self.drawEye(result=result)
        blitComponent(screen, comps["img_eyebrow_L"])
        blitComponent(screen, comps["img_eyebrow_R"])
        blitComponent(screen, comps["img_hair_front"])
