import time
import threading
import numpy as np
import cv2
from pubsub import pub
import mediapipe as mp
import numpy as np
from Const import *
from ValueUtils import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


def convert_blendshape_dict(raw_shape):
    # 【關鍵步驟】用 Comprehension 轉成字典 { 'jawOpen': 0.9, 'eyeBlink_L': 0.0 ... }
    blendshape_dict = {shape.category_name: shape.score for shape in raw_shape}
    return blendshape_dict


import cv2
import numpy as np


def get_reprojection_error(
    model_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
):
    """
    計算 MediaPipe 地標的重投影誤差，用於偵測遮擋導致的幾何畸變。
    (Comment: Calculate the reprojection error to detect facial distortion.)
    """
    projected_points, _ = cv2.projectPoints(
        model_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    # 計算平均每個點的像素距離差
    error = np.mean(np.linalg.norm(image_points - projected_points, axis=1))

    return error


class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True 是關鍵，這樣才會回傳瞳孔(Iris)的座標
        base_options = python.BaseOptions(
            model_asset_path="mediapipe_model/face_landmarker.task"
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_face_presence_confidence=0.8,
            min_tracking_confidence=0.7,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            result_callback=self.store_result,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        # self.face_mesh = self.mp_face_mesh.FaceMesh(
        #    max_num_faces=1,
        #    refine_landmarks=True,
        #    min_detection_confidence=0.5,
        #    min_tracking_confidence=0.5,
        # )
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

        # 相機矩陣 (之後在 process 裡初始化一次即可)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # 假設無鏡頭變形.VideoCapture(0)
        self.first = True
        self.results = None
        self.last_angle = (0, 0, 0)
        self.blendshapes = None

    def store_result(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        # Store result for main loop to access
        self.results = result
        if len(self.results.face_blendshapes) > 0:
            self.blendshapes = convert_blendshape_dict(self.results.face_blendshapes[0])
        return

    def process(self):
        success, image = self.cap.read()
        image.flags.writeable = False
        cv2.imshow("tracking result", image)
        cv2.waitKey(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.results = self.face_mesh.process(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        frame_timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, frame_timestamp_ms)
        self.image = image
        return image

    def get_iris_pos(self):
        """
        回傳眼球的相對位置 (x, y)
        x: -1.0 (左) ~ 1.0 (右), 0.0 是中間
        y: -1.0 (上) ~ 1.0 (下), 0.0 是中間
        """
        if self.blendshapes is None:
            return

        dx = self.blendshapes["eyeLookOutLeft"] - self.blendshapes["eyeLookInLeft"]
        dy = self.blendshapes["eyeLookUpLeft"] - self.blendshapes["eyeLookDownLeft"]
        return -dx, -dy * 1.2

    def calculate_mouth_openness(self):
        """
        計算嘴巴張開比例 (MAR - Mouth Aspect Ratio)

        Args:
            landmarks: MediaPipe 返回的 normalized_landmarks (包含 x, y)
            image_width: 畫布寬度 (用於還原座標)
            image_height: 畫布高度

        Returns:
            float: 原始 MAR 數值 (通常在 0.0 ~ 0.5 之間)
        """
        if self.blendshapes is None:
            return 0
        jawOpen = self.blendshapes["jawOpen"]
        jawOpen = map_range(jawOpen, 0, 0.3, 0, 1)
        return jawOpen

    def get_eye_blink_ratio(self):
        """
        計算左右眼的開闔程度 (Blink Ratio)
        回傳: (left_ratio, right_ratio)
        數值通常在 0.0 (閉) ~ 0.3 (大開) 之間
        """
        if self.blendshapes is None:
            return
        return (
            1 - self.blendshapes["eyeBlinkLeft"],
            1 - self.blendshapes["eyeBlinkRight"],
        )

    def get_head_pose(self, img_w, img_h):
        """
        計算頭部姿態 (Yaw, Pitch, Roll)
        回傳: yaw, pitch, roll (單位: 度 degree)
        """
        results = self.results
        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
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
            (face_landmarks[i].x * img_w, face_landmarks[i].y * img_h)
            for i in pose_landmark_index
        ]

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
        self.last_angle = yaw, pitch, roll

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
        for lm in face_landmarks:
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
            if self._tracker.results is None:
                continue
            yaw, pitch, roll = self._tracker.get_head_pose(width, height)
            bl, br = self._tracker.get_eye_blink_ratio()
            dx, dy = self._tracker.get_iris_pos()
            mo = self._tracker.calculate_mouth_openness()

            pub.sendMessage(
                "FaceInfo",
                face_info={
                    "PupilsPos": (dx, dy),
                    "Blinking": (bl, br),
                    "MouthOpenness": mo,
                    "Pose": (yaw, pitch, roll),
                },
            )
            # 稍微休息一下，避免吃光 CPU (約 60 FPS)
            time.sleep(0.016)

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self._tracker.release()


class FakeTracker:
    """
    非同步追蹤器：用獨立執行緒跑 OpenCV，
    確保主視窗被 Windows 卡住時 (例如拖曳視窗)，追蹤不會中斷。
    """

    def __init__(self):
        pub.sendMessage(
            "FaceInfo",
            face_info={
                "PupilsPos": (0, 0),
                "Blinking": (0, 0),
                "MouthOpenness": 0,
                "Pose": (0, 0, 0),
            },
        )

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        """這是背景執行緒在做的事：不斷更新數據"""
        while self.running:
            pub.sendMessage(
                "FaceInfo",
                face_info={
                    "PupilsPos": (0, 0),
                    "Blinking": (1, 1),
                    "MouthOpenness": 0,
                    "Pose": (0, 0, 0),
                },
            )
            # 稍微休息一下，避免吃光 CPU (約 60 FPS)
            time.sleep(0.016)

    def release(self):
        self.running = False
        return
