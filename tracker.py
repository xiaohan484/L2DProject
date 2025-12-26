import cv2
import mediapipe as mp
import numpy as np
import pygame

class FakeTracker:
    def __init__(self):
        return
    def isFake(self):
        return True
    def get_iris_pos(self):
        mouse_x, mouse_y = pygame.mouse.get_pos() # 暫時用滑鼠測試
        return mouse_x,mouse_y
    def release(self):
        return True

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True 是關鍵，這樣才會回傳瞳孔(Iris)的座標
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
    def isFake(self):
        return False
    def get_iris_pos(self):
        """
        回傳眼球的相對位置 (x, y)
        x: -1.0 (左) ~ 1.0 (右), 0.0 是中間
        y: -1.0 (上) ~ 1.0 (下), 0.0 是中間
        """
        success, image = self.cap.read()
        if not success:
            return 0, 0 # 讀不到畫面時回傳歸零

        # 效能優化：標記唯讀可以加速 MediaPipe 處理
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        dx, dy = 0, 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- 核心演算法：計算瞳孔在眼框中的相對位置 ---
            # 我們使用左眼來做基準 (MediaPipe 的左眼對應到畫面右邊的臉)
            # 索引: 33(眼頭), 133(眼尾), 468(瞳孔中心)
            
            # 取得座標 (只要 x, y)
            in_x, in_y = landmarks[33].x, landmarks[33].y   # Inner Corner
            out_x, out_y = landmarks[133].x, landmarks[133].y # Outer Corner
            iris_x, iris_y = landmarks[468].x, landmarks[468].y # Iris Center

            # 1. 計算眼寬和眼高 (大略估算)
            eye_width = out_x - in_x
            # 為了簡化，高度我們先用眼寬的一個比例來抓，或者暫時忽略 Y 軸精確度
            
            # 2. 計算瞳孔相對於眼頭的距離
            dist_x = iris_x - in_x
            
            # 3. 算出比例 (0.0 ~ 1.0)
            # 0.5 代表在中間。一般來說瞳孔不會碰到眼角，所以範圍大概在 0.3~0.7 之間
            ratio_x = dist_x / eye_width

            # 4. 正規化到 -1 ~ 1 (為了給 Pygame 用)
            # 我們假設 0.5 是中心點。
            # (ratio - 0.5) * 2 -> 變成 -1 ~ 1
            # 乘上一個敏感度係數 (Sensitivity)，讓眼睛動得更靈敏
            SENSITIVITY = 2.5 
            dx = (ratio_x - 0.45) * 2 * SENSITIVITY # 0.45 是稍微修正中心偏移
            
            # Y 軸簡單處理 (眼球上下)
            # 使用眼皮上下點：159 (上), 145 (下)
            top_y = landmarks[159].y
            bot_y = landmarks[145].y
            eye_height = bot_y - top_y
            dist_y = iris_y - top_y
            ratio_y = dist_y / eye_height
            dy = (ratio_y - 0.45) * 2 * SENSITIVITY

            # 限制數值在 -1 ~ 1 之間 (Clamping)
            dx = -max(-1.0, min(1.0, dx))
            dy = -max(-1.0, min(1.0, dy))

        return dx, dy

    def release(self):
        self.cap.release()