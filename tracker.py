import cv2
import mediapipe as mp
import numpy as np
import time
import threading

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
    def process(self):
        success, image = self.cap.read()
        image.flags.writeable = False
        cv2.imshow("tracking result", image)
        cv2.waitKey(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(image)
        return 
    def get_iris_pos(self):
        """
        回傳眼球的相對位置 (x, y)
        x: -1.0 (左) ~ 1.0 (右), 0.0 是中間
        y: -1.0 (上) ~ 1.0 (下), 0.0 是中間
        """
        results = self.results
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

    def get_eye_blink_ratio(self):
        """
        計算左右眼的開闔程度 (Blink Ratio)
        回傳: (left_ratio, right_ratio)
        數值通常在 0.0 (閉) ~ 0.3 (大開) 之間
        """
        results = self.results
        left_ratio = 1.0
        right_ratio = 1.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # --- 左眼關鍵點 (MediaPipe 的左眼對應畫面右側) ---
            # 垂直: 159 (上), 145 (下)
            # 水平: 33 (內), 133 (外)
            l_top = landmarks[159].y
            l_bot = landmarks[145].y
            l_in  = landmarks[33].x
            l_out = landmarks[133].x
            
            # 計算垂直距離 / 水平距離 (標準化，避免離鏡頭遠近影響數值)
            # 加上一個極小的數 1e-6 避免除以零
            left_ratio = abs(l_bot - l_top) / (abs(l_out - l_in) + 1e-6)

            # --- 右眼關鍵點 ---
            # 垂直: 386 (上), 374 (下)
            # 水平: 362 (內), 263 (外)
            r_top = landmarks[386].y
            r_bot = landmarks[374].y
            r_in  = landmarks[362].x
            r_out = landmarks[263].x
            
            right_ratio = abs(r_bot - r_top) / (abs(r_out - r_in) + 1e-6)

        return left_ratio, right_ratio

    def release(self):
        self.cap.release()
class AsyncFaceTracker:
    """
    非同步追蹤器：用獨立執行緒跑 OpenCV，
    確保主視窗被 Windows 卡住時 (例如拖曳視窗)，追蹤不會中斷。
    """
    def __init__(self):
        # 建立原本的 Tracker
        self._tracker = FaceTracker()
        
        # 共享變數 (加上 Lock 避免讀寫衝突，雖然 Python GIL 某種程度上會保護)
        self.lock = threading.Lock()
        self._current_iris_pos = (0.0, 0.0)
        self._current_blink_ratio = (1.0, 1.0)
        self._current_head_pose = (0.0, 0.0)
        
        self.running = True
        
        # 建立並啟動執行緒
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        """這是背景執行緒在做的事：不斷更新數據"""
        while self.running:
            self._tracker.process()
            # 1. 取得數據 (這一步最耗時，現在不會卡住 UI 了)
            dx, dy = self._tracker.get_iris_pos()
            bl, br = self._tracker.get_eye_blink_ratio()
            #yaw, pitch = self._tracker.get_head_pose()
            
            # 2. 存入共享變數
            with self.lock:
                self._current_iris_pos = (dx, dy)
                self._current_blink_ratio = (bl, br)
                #self._current_head_pose = (yaw, pitch)
            
            # 稍微休息一下，避免吃光 CPU (約 60 FPS)
            time.sleep(0.016)

    # --- 外部呼叫的介面 (讀取最新值) ---
    # 這些函式會由 Main Thread (UI) 呼叫，速度極快，因為只是讀變數
    
    def get_iris_pos(self):
        with self.lock:
            return self._current_iris_pos

    def get_eye_blink_ratio(self):
        with self.lock:
            return self._current_blink_ratio

    def get_head_pose(self):
        with self.lock:
            return self._current_head_pose

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self._tracker.release()