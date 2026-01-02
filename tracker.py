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
        
        # --- [新增] Head Pose Estimation 需要的參數 ---
        self.img_w = 640 # 預設，之後會動態更新
        self.img_h = 480
        
        # 定義標準 3D 臉部模型的 6 個特徵點 (世界座標)
        # 順序：鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (原點)
            (0.0, 330.0, 65.0),          # Chin (下巴在下面 -> Y是正的)
            (-225.0, -170.0, 135.0),     # Left eye left corner (眼睛在上面 -> Y是負的)
            (225.0, -170.0, 135.0),      # Right eye right corner
            (-150.0, 150.0, 125.0),      # Left Mouth corner (嘴巴在下面 -> Y是正的)
            (150.0, 150.0, 125.0),        # Right Mouth corner

            ## --- [新增] 穩定錨點 ---
            #(-60.0, -170.0, 100.0),      # 7. Left Eye Inner Corner (內眼角) - 靠鼻樑
            #(60.0, -170.0, 100.0),       # 8. Right Eye Inner Corner (內眼角) - 靠鼻樑

            ## --- [修正] 9, 10 眉峰/額頭 (位置較高，且稍微深一點) ---
            ## Y 設為 300.0 (比眼睛 170 高)
            ## X 設為 180.0 (寬度適中)
            #(-180.0, -300.0, 100.0),     # 9. 左眉峰 (對應 105)
            #(180.0, -300.0, 100.0)       # 10. 右眉峰 (對應 334)
        ], dtype=np.float64)
        
        # 相機矩陣 (之後在 process 裡初始化一次即可)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1)) # 假設無鏡頭變形.VideoCapture(0)
    def isFake(self):
        return False
    def process(self):
        success, image = self.cap.read()
        image.flags.writeable = False
        cv2.imshow("tracking result", image)
        cv2.waitKey(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(image)
        self.image = image
        return image
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
    def get_head_pose(self, img_w, img_h):
        """
        計算頭部姿態 (Yaw, Pitch, Roll)
        回傳: yaw, pitch, roll (單位: 度 degree)
        """
        results = self.results
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
        else:
            return (0,0,0)
        self.img_w = img_w
        self.img_h = img_h

        # 如果還沒設定相機矩陣，設一個估計值
        if self.cam_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            self.cam_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

        # 1. 從 MediaPipe 提取對應的 6 個 2D 關鍵點
        # 注意：MediaPipe 的點是正規化的 (0~1)，要乘上寬高
        # Index: Nose=1, Chin=152, L_Eye=33, R_Eye=263, L_Mouth=61, R_Mouth=291
    
        image_points = np.array([
            (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),     # Nose tip
            (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h), # Chin
            (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),   # Left Eye
            (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h), # Right Eye
            (face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h),   # Left Mouth
            (face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h),  # Right Mouth
            # --- [新增] ---
            #(face_landmarks.landmark[362].x * img_w, face_landmarks.landmark[362].y * img_h), # 7. 左內眼角
            #(face_landmarks.landmark[133].x * img_w, face_landmarks.landmark[133].y * img_h), # 8. 右內眼角
            #(face_landmarks.landmark[105].x * img_w, face_landmarks.landmark[105].y * img_h),  # 9. 左眉尾 (這點相對穩定)
            #(face_landmarks.landmark[334].x * img_w, face_landmarks.landmark[334].y * img_h)  # 10. 右眉尾
        ], dtype="double")

        # 2. 呼叫 SolvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.cam_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_SQPNP
        )

        # 3. 將旋轉向量轉換為歐拉角 (Euler Angles)
        #這部分數學比較深，主要是把旋轉矩陣轉成我們看得懂的角度
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # 4. 提取 Pitch, Yaw, Roll
        # 根據 OpenCV 的座標系定義：
        # angles[0] = Pitch (抬頭低頭)
        # angles[1] = Yaw (左右轉)
        # angles[2] = Roll (歪頭)
    
        pitch = angles[0]  # 轉換比例微調 (視需求調整強度)
        yaw   = angles[1]
        roll  = angles[2] 
        #pitch = max(-90, min(90, pitch))
        #yaw   = max(-90, min(90, yaw))
        #roll  = max(-90, min(90, roll))

        # 這裡的數值通常很敏感，可能需要限制範圍 (Clamp)
        # 顯示視窗
        debug = True
        if debug:
            debug_board = np.zeros((480, 640, 3), dtype=np.uint8)
            frontal_points = self.get_frontal_landmarks(face_landmarks,rmat,img_w,img_h)
            for point in frontal_points:
                cv2.circle(debug_board, point, 1, (0, 255, 0), -1) # 畫綠色小點
            # 畫出鼻尖 (紅色大點) 作為參考中心
            if len(frontal_points) > 1:
                 cv2.circle(debug_board, frontal_points[1], 3, (0, 0, 255), -1)
                 cv2.circle(debug_board, frontal_points[152], 3, (0, 0, 255), -1)
                 cv2.circle(debug_board, frontal_points[33], 3, (0, 0, 255), -1)
                 cv2.circle(debug_board, frontal_points[263], 3, (0, 0, 255), -1)
                 cv2.circle(debug_board, frontal_points[61], 3, (0, 0, 255), -1)
                 cv2.circle(debug_board, frontal_points[291], 3, (0, 0, 255), -1)

                 #cv2.circle(debug_board, frontal_points[362], 3, (0, 0, 255), -1)
                 #cv2.circle(debug_board, frontal_points[133], 3, (0, 0, 255), -1)
                 #cv2.circle(debug_board, frontal_points[105], 3, (0, 0, 255), -1)
                 #cv2.circle(debug_board, frontal_points[334], 3, (0, 0, 255), -1)

            # 顯示視窗
            cv2.imshow("Debug: Frontalized View", debug_board)
            cv2.waitKey(1)
        return yaw, pitch, roll
    def get_frontal_landmarks(self, face_landmarks,rmat, img_w, img_h):
        """
        輸入：當前歪斜的 landmarks 和旋轉向量 rvec
        輸出：被「轉正」後的 2D landmarks 座標列表 (用於繪圖)
        """
        # 1. 取得旋轉矩陣 R
        R = rmat
        
        # 2. 計算逆向旋轉矩陣 (轉置矩陣)
        # 這個矩陣的作用是把歪的頭轉正
        R_inv = R.T

        # 3. 收集當前所有 landmarks 的 3D 座標
        # MediaPipe 提供的 z 座標是相對深度，我們需要把它變成類似像素的單位
        landmarks_3d_list = []
        for lm in face_landmarks.landmark:
            # 將標準化座標轉換為近似的 3D 空間座標
            # x, y 乘上寬高，z 也乘上寬度作為深度比例估計
            lx, ly, lz = lm.x * img_w, lm.y * img_h, lm.z * img_w
            landmarks_3d_list.append([lx, ly, lz])
            
        points_np = np.array(landmarks_3d_list, dtype=np.float32)

        # 4. 【關鍵步驟】將座標中心化
        # 旋轉是繞著原點(0,0,0)轉的。我們必須把鼻尖移到原點。
        nose_tip_index = 1
        nose_pos = points_np[nose_tip_index]
        centered_points = points_np - nose_pos

        # 5. 【核心數學】應用逆向旋轉
        # 矩陣乘法： Unrotated_P = R_inv * Centered_P
        # 因為我們的 points 是 N x 3 的形狀，要轉置一下才能相乘
        # 結果形狀是 3 x N
        #unrotated_points_3d_T = R_inv @ centered_points.T
        # 轉回來變成 N x 3
        #unrotated_points_3d = unrotated_points_3d_T.T
        unrotated_points_3d = centered_points

        # 6. 將轉正後的 3D 點投影回 2D 畫面以便繪製
        frontal_points_2d = []
        # 設定繪製畫面的中心點 (例如放在畫面中間)
        draw_center_x, draw_center_y = img_w // 2, img_h // 2

        # A. 找出轉正後的人臉邊界 (Bounding Box)
        # unrotated_points_3d[:, 0] 是所有點的 X
        # unrotated_points_3d[:, 1] 是所有點的 Y
        min_x = np.min(unrotated_points_3d[:, 0])
        max_x = np.max(unrotated_points_3d[:, 0])
        min_y = np.min(unrotated_points_3d[:, 1])
        max_y = np.max(unrotated_points_3d[:, 1])

        face_w = max_x - min_x
        face_h = max_y - min_y
        
        # B. 計算縮放比例 (Scale)
        # 我們希望臉的寬度佔畫面的 padding_ratio
        # 或者是高度佔畫面的 padding_ratio
        # 取兩者中較小的值，確保不會超出畫面
        padding_ratio = 0.8
        scale_x = (img_w * padding_ratio) / face_w
        scale_y = (img_h * padding_ratio) / face_h
        final_scale = min(scale_x, scale_y)
        # C. 計算中心偏移 (Centering)
        # 我們要把臉的 "幾何中心" 移到 "視窗中心"
        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2
        
        window_center_x = img_w // 2
        window_center_y = img_h // 2
        
        frontal_points_2d = []
        for p3d in unrotated_points_3d:
            px = int((p3d[0] - face_center_x) * final_scale + window_center_x)
            py = int((p3d[1] - face_center_y) * final_scale + window_center_y)
            frontal_points_2d.append((px, py))
            
        return frontal_points_2d

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
            img = self._tracker.process()
            height, width, channels = img.shape
            # 1. 取得數據 (這一步最耗時，現在不會卡住 UI 了)
            dx, dy = self._tracker.get_iris_pos()
            bl, br = self._tracker.get_eye_blink_ratio()
            yaw, pitch,roll = self._tracker.get_head_pose(width,height)

            
            # 2. 存入共享變數
            with self.lock:
                self._current_iris_pos = (dx, dy)
                self._current_blink_ratio = (bl, br)
                self._current_head_pose = (yaw, pitch,roll)
            
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