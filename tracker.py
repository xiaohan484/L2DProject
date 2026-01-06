import time
import threading
import numpy as np
import cv2
from pubsub import pub
import mediapipe as mp
import numpy as np
from Const import *
from ValueUtils import *

# pose_landmark_index = [1,152,33,263,61,291]
pose_landmark_index = [
    1,
    151,
    101,
    330,
    345,
    116,
    103,
    332,
    156,
    383,
    195,
    168,
    322,
    165,
    69,
    299,
]


def load_personal_model(json_path):
    """
    從 JSON 載入個人化臉模，並提取指定的剛性特徵點。

    Args:
        json_path (str): 剛剛存下來的 json 檔案路徑
        target_indices (dict or list): 你需要的剛性點 Index (例如 { 'NOSE': 1, ... })

    Returns:
        np.array: 给 solvePnP 用的 model_points (N, 3)
    """
    try:
        print(f"📂 Loading personal model from {json_path}...")
        with open(json_path, "r") as f:
            all_landmarks = json.load(f)

        selected_points = []

        for pt in all_landmarks:
            selected_points.append(pt)

        # 轉成 NumPy Float64 (solvePnP 必要格式)
        model_points_np = np.array(selected_points, dtype=np.float64)

        print(f"✅ Loaded {len(model_points_np)} rigid points successfully.")
        return model_points_np

    except FileNotFoundError:
        print(f"❌ Error: File {json_path} not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


my_face = load_personal_model("assets/privacy/my_personal_landmarks.json")


def project_perspective_auto_fit(
    points_3d, win_w, win_h, fov_degrees=60.0, padding=0.1
):
    """
    透視投影 + 自動縮放 (Auto-Dolly)

    1. 將 3D 點雲移到原點 (0,0,0)
    2. 根據 FOV 計算虛擬焦距
    3. 計算需要將物體推遠多少距離 (tz)，才能剛好填滿視窗
    4. 執行透視投影: u = fx * (x / z) + cx

    Args:
        points_3d (np.array): (N, 3) 3D 點
        win_w, win_h (int): 視窗大小
        fov_degrees (float): 虛擬相機的視野角度 (預設 60 度，接近人眼/Webcam)
        padding (float): 邊緣留白比例 (0.1 = 10%)

    Returns:
        np.array: (N, 2) 整數座標 (u, v)
    """
    # 1. 複製資料以免改到原始數據
    p3d = points_3d.copy()

    # 2. 幾何中心歸零 (Centering)
    # 找出物體中心，並將所有點移到以 (0,0,0) 為中心
    center = np.mean(p3d, axis=0)
    p3d_centered = p3d - center

    # 3. 計算 3D 空間中的包圍盒尺寸 (Bounding Box Size)
    min_xyz = np.min(p3d_centered, axis=0)
    max_xyz = np.max(p3d_centered, axis=0)
    face_w_3d = max_xyz[0] - min_xyz[0]
    face_h_3d = max_xyz[1] - min_xyz[1]

    # 4. 建立虛擬相機參數 (Intrinsics)
    # fx = (W / 2) / tan(fov / 2)
    fov_rad = np.radians(fov_degrees)
    focal_length = (win_w / 2) / np.tan(fov_rad / 2)

    cx = win_w / 2
    cy = win_h / 2

    # 5. 計算「推軌距離」 (Dolly Distance / Z-Offset)
    # 我們需要多深的 Z，才能讓投影後的寬度等於視窗寬度？
    # 公式導出: projected_size = focal_length * (real_size / depth)
    # 所以: depth = focal_length * real_size / projected_size

    target_w = win_w * (1.0 - padding * 2)
    target_h = win_h * (1.0 - padding * 2)

    # 避免除以零
    face_w_3d = max(face_w_3d, 1e-5)
    face_h_3d = max(face_h_3d, 1e-5)

    dist_w = focal_length * (face_w_3d / target_w)
    dist_h = focal_length * (face_h_3d / target_h)

    # 取最遠的距離，確保寬和高都塞得進去
    z_offset = max(dist_w, dist_h)

    # 稍微加一點近平面剪裁保護 (Near Plane Clipping protection)
    # 讓臉的最凸點不會戳到鏡頭後面
    max_z_variation = np.max(np.abs(p3d_centered[:, 2]))
    z_dist = z_offset + max_z_variation + 10  # 安全距離

    # 6. 執行透視投影 (Perspective Projection)
    # u = fx * x / z + cx
    # v = fy * y / z + cy

    # 加上深度推移
    z_coords = p3d_centered[:, 2] + z_dist

    # 投影運算
    # 注意: 如果你的 Y 軸方向跟螢幕相反，可以在這裡加負號 (-p3d_centered[:, 1])
    # 這裡假設 Y 向下為正 (OpenCV 標準)
    u = (focal_length * p3d_centered[:, 0] / z_coords) + cx
    v = (focal_length * p3d_centered[:, 1] / z_coords) + cy

    return np.stack((u, v), axis=1).astype(np.int32)


def fit_points_to_window(uv_points, window_w, window_h, padding_ratio=0.1):
    """
    接收任意一組 UV 點 (N, 2)，自動縮放平移以填滿視窗。

    Args:
        uv_points (np.array): 形狀 (N, 2) 的點，單位不限 (可以是 0~1 或任意 pixel)
        window_w (int): 目標視窗寬度
        window_h (int): 目標視窗高度
        padding_ratio (float): 邊緣留白比例 (例如 0.1 代表留 10% 空隙)

    Returns:
        np.array: 轉換後的 (N, 2) 整數座標，可直接畫圖
    """
    points = np.array(uv_points)

    # 1. 找出目前的邊界 (Bounding Box)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    current_w = max_x - min_x
    current_h = max_y - min_y
    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2

    # 2. 避免除以零 (如果只有一個點或點重疊)
    current_w = max(current_w, 1e-5)
    current_h = max(current_h, 1e-5)

    # 3. 計算縮放比例 (Scale to Fit)
    # 我們希望寬度能撐滿視窗，高度也能撐滿視窗
    # 但為了不變形，我們取 "兩者中較小" 的縮放倍率
    scale_x = window_w / current_w
    scale_y = window_h / current_h

    # 實際縮放倍率 (乘上 (1 - padding*2) 是為了留出雙邊的白邊)
    final_scale = min(scale_x, scale_y) * (1.0 - padding_ratio * 2)

    # 4. 執行變換公式
    # New = (Old - Old_Center) * Scale + New_Window_Center

    target_center_x = window_w // 2
    target_center_y = window_h // 2

    new_xs = (points[:, 0] - current_center_x) * final_scale + target_center_x
    new_ys = (points[:, 1] - current_center_y) * final_scale + target_center_y

    # 5. 組合並轉整數
    new_points = np.stack((new_xs, new_ys), axis=1).astype(np.int32)

    return new_points


class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True 是關鍵，這樣才會回傳瞳孔(Iris)的座標
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(0)

        # --- [新增] Head Pose Estimation 需要的參數 ---
        self.img_w = 640  # 預設，之後會動態更新
        self.img_h = 480

        # 定義標準 3D 臉部模型的 6 個特徵點 (世界座標)
        # 順序：鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角
        points = [
            (my_face[i][0], my_face[i][1], my_face[i][2]) for i in pose_landmark_index
        ]
        self.model_points = np.array(points, dtype=np.float64)
        # self.model_points = np.array([
        #    (0.0, 0.0, 0.0),             # Nose tip (原點)
        #    (0.0, 330.0, 65.0),          # Chin (下巴在下面 -> Y是正的)
        #    (-225.0, -170.0, 135.0),     # Left eye left corner (眼睛在上面 -> Y是負的)
        #    (225.0, -170.0, 135.0),      # Right eye right corner
        #    (-150.0, 150.0, 125.0),      # Left Mouth corner (嘴巴在下面 -> Y是正的)
        #    (150.0, 150.0, 125.0),        # Right Mouth corner

        #    ## --- [新增] 穩定錨點 ---
        #    #(-60.0, -170.0, 100.0),      # 7. Left Eye Inner Corner (內眼角) - 靠鼻樑
        #    #(60.0, -170.0, 100.0),       # 8. Right Eye Inner Corner (內眼角) - 靠鼻樑

        #    ## --- [修正] 9, 10 眉峰/額頭 (位置較高，且稍微深一點) ---
        #    ## Y 設為 300.0 (比眼睛 170 高)
        #    ## X 設為 180.0 (寬度適中)
        #    #(-180.0, -300.0, 100.0),     # 9. 左眉峰 (對應 105)
        #    #(180.0, -300.0, 100.0)       # 10. 右眉峰 (對應 334)
        # ], dtype=np.float64)

        # 相機矩陣 (之後在 process 裡初始化一次即可)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # 假設無鏡頭變形.VideoCapture(0)
        self.first = True

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
            in_x, in_y = landmarks[33].x, landmarks[33].y  # Inner Corner
            out_x, out_y = landmarks[133].x, landmarks[133].y  # Outer Corner
            iris_x, iris_y = landmarks[468].x, landmarks[468].y  # Iris Center

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
            dx = (ratio_x - 0.45) * 2 * SENSITIVITY  # 0.45 是稍微修正中心偏移

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

    def calculate_mouth_openness(self, image_width, image_height):
        """
        計算嘴巴張開比例 (MAR - Mouth Aspect Ratio)

        Args:
            landmarks: MediaPipe 返回的 normalized_landmarks (包含 x, y)
            image_width: 畫布寬度 (用於還原座標)
            image_height: 畫布高度

        Returns:
            float: 原始 MAR 數值 (通常在 0.0 ~ 0.5 之間)
        """
        results = self.results
        if results.multi_face_landmarks is None:
            return
        landmarks = results.multi_face_landmarks[0].landmark
        dx, dy = 0, 0
        # 1. 定義關鍵點索引 (Inner Lips)
        IDX_TOP = 13
        IDX_BOTTOM = 14
        IDX_LEFT = 61
        IDX_RIGHT = 291
        # 2. 取得座標點 (將 Normalized 0~1 轉為 Pixel 座標)
        # 注意：如果不轉 Pixel 直接用 Normalized 算也可以，但轉 Pixel 比較直觀好除錯
        p_top = np.array(
            [landmarks[IDX_TOP].x * image_width, landmarks[IDX_TOP].y * image_height]
        )
        p_bottom = np.array(
            [
                landmarks[IDX_BOTTOM].x * image_width,
                landmarks[IDX_BOTTOM].y * image_height,
            ]
        )
        p_left = np.array(
            [landmarks[IDX_LEFT].x * image_width, landmarks[IDX_LEFT].y * image_height]
        )
        p_right = np.array(
            [
                landmarks[IDX_RIGHT].x * image_width,
                landmarks[IDX_RIGHT].y * image_height,
            ]
        )

        # 3. 計算歐幾里得距離 (Euclidean Distance)
        # 垂直距離 (開合程度)
        height = np.linalg.norm(p_top - p_bottom)

        # 水平距離 (嘴巴寬度 - 作為分母)
        width = np.linalg.norm(p_left - p_right)

        # 4. 防呆機制 (避免除以零)
        if width < 1e-6:
            return 0.0

        # 5. 計算比例
        mar = height / width
        mar = map_range(mar, 0, 0.5, 0, 1)
        return mar

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
            l_in = landmarks[33].x
            l_out = landmarks[133].x

            # 計算垂直距離 / 水平距離 (標準化，避免離鏡頭遠近影響數值)
            # 加上一個極小的數 1e-6 避免除以零
            left_ratio = abs(l_bot - l_top) / (abs(l_out - l_in) + 1e-6)

            # --- 右眼關鍵點 ---
            # 垂直: 386 (上), 374 (下)
            # 水平: 362 (內), 263 (外)
            r_top = landmarks[386].y
            r_bot = landmarks[374].y
            r_in = landmarks[362].x
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
            return (0, 0, 0)
        self.img_w = img_w
        self.img_h = img_h

        # 如果還沒設定相機矩陣，設一個估計值
        if self.cam_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            self.cam_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double",
            )

        # 1. 從 MediaPipe 提取對應的 6 個 2D 關鍵點
        # 注意：MediaPipe 的點是正規化的 (0~1)，要乘上寬高
        # Index: Nose=1, Chin=152, L_Eye=33, R_Eye=263, L_Mouth=61, R_Mouth=291
        points = [
            (face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h)
            for i in pose_landmark_index
        ]

        # [
        #    (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),     # Nose tip
        #    (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h), # Chin
        #    (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),   # Left Eye
        #    (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h), # Right Eye
        #    (face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h),   # Left Mouth
        #    (face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h),  # Right Mouth
        #    # --- [新增] ---
        #    #(face_landmarks.landmark[362].x * img_w, face_landmarks.landmark[362].y * img_h), # 7. 左內眼角
        #    #(face_landmarks.landmark[133].x * img_w, face_landmarks.landmark[133].y * img_h), # 8. 右內眼角
        #    #(face_landmarks.landmark[105].x * img_w, face_landmarks.landmark[105].y * img_h),  # 9. 左眉尾 (這點相對穩定)
        #    #(face_landmarks.landmark[334].x * img_w, face_landmarks.landmark[334].y * img_h)  # 10. 右眉尾
        #    ]
        image_points = np.array(points, dtype="double")

        # 2. 呼叫 SolvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.cam_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        # 3. 將旋轉向量轉換為歐拉角 (Euler Angles)
        # 這部分數學比較深，主要是把旋轉矩陣轉成我們看得懂的角度
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # 4. 提取 Pitch, Yaw, Roll
        # 根據 OpenCV 的座標系定義：
        # angles[0] = Pitch (抬頭低頭)
        # angles[1] = Yaw (左右轉)
        # angles[2] = Roll (歪頭)

        pitch = angles[0]  # 轉換比例微調 (視需求調整強度)
        yaw = angles[1]
        roll = angles[2]
        if self.first:
            self.first = False
            self.init_angle = yaw, pitch, roll
        iyaw, ipitch, iroll = self.init_angle
        pitch -= ipitch
        yaw -= iyaw
        roll -= iroll

        # pitch = max(-90, min(90, pitch))
        # yaw   = max(-90, min(90, yaw))
        # roll  = max(-90, min(90, roll))
        # 這裡的數值通常很敏感，可能需要限制範圍 (Clamp)
        # 顯示視窗
        debug = True
        if debug:
            debug_board = np.zeros((480, 640, 3), dtype=np.uint8)
            frontal_points = self.get_frontal_landmarks(
                rmat, face_landmarks, img_w, img_h
            )
            for point in frontal_points:
                cv2.circle(debug_board, point, 1, (0, 255, 0), -1)  # 畫綠色小點
            # 畫出鼻尖 (紅色大點) 作為參考中心
            if len(frontal_points) > 1:
                for i in pose_landmark_index:
                    cv2.circle(debug_board, frontal_points[i], 3, (0, 0, 255), -1)
            cv2.imshow("Debug: Frontalized View", debug_board)
            # 顯示視窗
            cv2.waitKey(1)
        return yaw, pitch, roll

    def get_frontal_landmarks(self, rmat, face_landmarks, img_w, img_h):
        """
        輸入：當前歪斜的 landmarks 和旋轉向量 rvec
        輸出：被「轉正」後的 2D landmarks 座標列表 (用於繪圖)
        """
        # 1. 取得旋轉矩陣 R
        # R = self.rt
        # R = self.rt

        # R = self.rt
        # 2. 計算逆向旋轉矩陣 (轉置矩陣)
        # 這個矩陣的作用是把歪的頭轉正
        # R_inv = np.eye(3)

        # 3. 收集當前所有 landmarks 的 3D 座標
        # MediaPipe 提供的 z 座標是相對深度，我們需要把它變成類似像素的單位
        landmarks_3d_list = []
        for lm in face_landmarks.landmark:
            # 將標準化座標轉換為近似的 3D 空間座標
            # x, y 乘上寬高，z 也乘上寬度作為深度比例估計
            lx, ly, lz = lm.x * img_w, lm.y * img_h, lm.z * img_w
            landmarks_3d_list.append([lx, ly, lz])

        points_np = np.array(landmarks_3d_list, dtype=np.float32)
        unrotated_points_3d = points_np
        frontal_points_2d = []

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
            yaw, pitch, roll = self._tracker.get_head_pose(width, height)

            bl, br = self._tracker.get_eye_blink_ratio()
            dx, dy = self._tracker.get_iris_pos()
            mo = self._tracker.calculate_mouth_openness(width, height)

            # 2. 存入共享變數
            with self.lock:
                pub.sendMessage("MouthOpenness", openness_value=mo)
                pub.sendMessage(
                    "EyeInfo", info={"Pupils_Pos": (dx, dy), "Blinking": (bl, br)}
                )

                self._current_iris_pos = (dx, dy)
                self._current_blink_ratio = (bl, br)
                self._current_head_pose = (yaw, pitch, roll)
                self._current_mouth_openness = mo

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

    def get_mouth_openness(self):
        with self.lock:
            return self._current_mouth_openness

    def get_head_pose(self):
        with self.lock:
            return self._current_head_pose

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self._tracker.release()
