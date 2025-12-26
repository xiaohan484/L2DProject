import pygame
import os
import math

CAM_X = 350
CAM_Y = 100
CAM_W = 600
CAM_H = 500
FPS = 30

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


class Face2DDraw:
    def __init__(self, isFake: bool, screen):
        self.isFake = isFake
        self.screen = screen
        self.debug_font = pygame.font.SysFont("Arial", 24) # 或 "Consolas"
        self.components = {
            "img_face"        : load_img("Face.png"),
            "img_eye_white_L" : load_img("EyeWhiteL.png"),
            "img_eye_white_R" : load_img("EyeWhiteR.png"),
            "img_eye_pupil_L" : load_img("EyePupilL.png"),
            "img_eye_pupil_R" : load_img("EyePupilR.png"),
            "img_eye_lash_L"  : load_img("EyeLashL.png"),
            "img_eye_lash_R"  : load_img("EyeLashR.png"),
            "img_eyebrow_L"   : load_img("EyeBrowL.png"),
            "img_eyebrow_R"   : load_img("EyeBrowR.png"),
            "img_hair_front"  : load_img("HairFront.png")
        }
    def draw(self, iris_pos, calibration):
        calibration_x,calibration_y = calibration
        #mouse_x, mouse_y = pygame.mouse.get_pos() # 暫時用滑鼠測試
        target_dx, target_dy = iris_pos
        target_dx -= calibration_x
        target_dy -= calibration_y
        offset_x,offset_y = evaluatePupilOffset(target_dx, target_dy,self.isFake)

        for pattern,comp in self.components.items():
            if "pupil" in pattern:
                if pattern == "img_eye_pupil_L":
                    draw_eye_masked(self.screen, self.components["img_eye_white_L"], \
                                    comp, (offset_x,offset_y))
                elif pattern == "img_eye_pupil_R":
                    draw_eye_masked(self.screen, self.components["img_eye_white_R"], \
                                    comp, (offset_x,offset_y))
            else:
                blitComponent(self.screen, comp)
        debug_text = f"X: {target_dx:.2f}| {calibration_x:.2f}  Y: {target_dy:.2f}"
        color = (255, 0, 0) if (target_dx == 0 and target_dy == 0) else (0, 0, 255)
        text_surf = self.debug_font.render(debug_text, True, color)
        help_text = self.debug_font.render("Press 'C' to Center Eyes", True, (50, 50, 50))
        self.screen.blit(text_surf, (10, 10)) # 畫在左上角
        self.screen.blit(help_text, (10, 40))