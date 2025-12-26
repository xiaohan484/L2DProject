import pygame
from tracker import FaceTracker,FakeTracker
from Face2DDraw import Face2DDraw

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

tracker = FakeTracker() #FaceTracker()
face_drawer = Face2DDraw(isFake=tracker.isFake,screen=screen)

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
                    raw_dx, raw_dy = tracker.get_iris_pos()
                    global calibration_x,calibration_y
                    calibration_x = raw_dx
                    calibration_y = raw_dy
                    print(f"已校正中心點: ({calibration_x:.2f}, {calibration_y:.2f})")
                
            screen.fill((0, 255, 0)) # 填滿綠色背景 (Green Screen)
            func((calibration_x,calibration_y))
            pygame.display.flip()
            clock.tick(FPS)
    return wrapper

@loop
def refresh(calibration):
    face_drawer.draw(tracker.get_iris_pos(), calibration)

# --- 主迴圈 ---
refresh()
tracker.release()
pygame.quit()