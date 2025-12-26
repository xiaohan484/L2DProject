import pygame
from tracker import FaceTracker,FakeTracker
from Face2DDraw import Face2DDraw, TrackingResult

# --- 設定區 ---
CAM_X = 350
CAM_Y = 100
CAM_W = 600
CAM_H = 500
FPS = 30
# --- 變數初始化 ---
calibration_x = 0.0
calibration_y = 0.0
SMOOTHING = 0.2  # 0.0 ~ 1.0，越小越滑順但延遲越高

# --- 初始化 Pygame ---
pygame.init()
pygame.font.init()
pygame.display.set_caption("My VTuber v0.1")
screen = pygame.display.set_mode((CAM_W, CAM_H))
clock = pygame.time.Clock()

#tracker = FakeTracker() #FaceTracker()
tracker = FaceTracker() #FaceTracker()
face_drawer = Face2DDraw(isFake=tracker.isFake(),screen=screen)

def loop(func):
    def wrapper():
        running = True
        while running:
            # 1. 處理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    # 取得當下原始的眼睛位置，設為新的零點
                    face_drawer.setCalib(tracker.get_iris_pos())
                
            screen.fill((0, 255, 0)) # 填滿綠色背景 (Green Screen)
            func()
            pygame.display.flip()
            clock.tick(FPS)
    return wrapper

import cv2
@loop
def refresh():
    tracker.process()

    result = TrackingResult()
    result.pupilsX ,result.pupilsY = tracker.get_iris_pos()
    result.blinkL ,result.blinkR = tracker.get_eye_blink_ratio()
    face_drawer.draw(result)

# --- 主迴圈 ---
refresh()
tracker.release()
pygame.quit()