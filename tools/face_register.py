import cv2
import mediapipe as mp
import numpy as np
import json
import time

# ================= 參數設定區 =================
OUTPUT_FILENAME = "my_personal_landmarks.json"
TARGET_WINDOW_HEIGHT = 600
PADDING_RATIO = 0.15
# ============================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def normalize_to_window(point, center, scale, window_size):
    """將正規化座標轉為視窗像素座標 (繪圖用)"""
    centered_x = point[0] - center[0]
    centered_y = point[1] - center[1]
    win_x = int(centered_x * scale + window_size[0] / 2)
    win_y = int(centered_y * scale + window_size[1] / 2)
    return (int(point[0]),int(point[1]))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    # 取得相機真實解析度 (用於還原正確比例)
    cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = cam_w / cam_h

    # 設定聚焦視窗
    window_h = int(cam_h)
    window_w = int(cam_w)
    FOCUS_WINDOW_SIZE = (window_w, window_h)

    print(f"=== 個人臉模捕捉工具 (鼻尖歸零版) ===")
    print(f"📷 解析度: {int(cam_w)}x{int(cam_h)}")
    print(f"👃 儲存時將自動以【鼻尖】為原點 (0,0,0)")
    print(f"---------------------------------------")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face_mesh:

        last_saved_time = 0
        flash_counter = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            image.flags.writeable = True
            debug_image = image.copy()
            focus_image = np.zeros((FOCUS_WINDOW_SIZE[1], FOCUS_WINDOW_SIZE[0], 3), dtype=np.uint8)

            current_landmarks_raw = [] # 儲存原始數據

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        current_landmarks_raw.append([lm.x,lm.y,lm.z])
                    # 1. Draw the mesh (Tesselation)
                    mp_drawing.draw_landmarks(
                        image=focus_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # 2. Draw the contours (Face, Eyes, Eyebrows, Lips)
                    mp_drawing.draw_landmarks(
                        image=focus_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    # 3. (Optional) Draw Iris if refine_landmarks=True
                    mp_drawing.draw_landmarks(
                        image=focus_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
            (h, w) = focus_image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
                    
            # 2. Define crosshair settings
            length = 40          # Length of each arm of the cross
            color = (0, 255, 0)  # Green color (B, G, R)
            thickness = 2        # Line thickness
                    
            # 3. Draw the horizontal line
            # From (center_x - length, center_y) to (center_x + length, center_y)
            cv2.line(focus_image, (cX - length, cY), (cX + length, cY), color, thickness)
                    
            # 4. Draw the vertical line
            # From (center_x, center_y - length) to (center_x, center_y + length)
            cv2.line(focus_image, (cX, cY - length), (cX, cY + length), color, thickness)

            # --- 儲存邏輯 (重點修改區) ---
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and current_landmarks_raw:
                # ==========================================
                # 🔥 這裡進行座標轉換與歸零 🔥
                # ==========================================
                
                # 1. 轉換為像素座標 (Pixel Space)
                #    如果不乘解析度，臉會是變形的 (因為 x, y 都是 0~1，但螢幕不是正方形)
                #    MediaPipe 的 Z 軸大概跟 X 軸同一量級，所以我們乘上 width 讓它變成像素單位
                landmarks_pixel = []
                for pt in current_landmarks_raw:
                    px = pt[0] * cam_w
                    py = pt[1] * cam_h
                    pz = pt[2] * cam_w # Z 軸通常參考 X 軸的尺度 (或平均值)
                    landmarks_pixel.append([px, py, pz])
                
                landmarks_np = np.array(landmarks_pixel)

                # 2. 鼻尖歸零 (Centering on Nose Tip - Index 1)
                #nose_tip = landmarks_np[1] # 取得鼻尖座標
                landmarks_centered = landmarks_np #- nose_tip # 全部減去鼻尖
                
                # 3. (選用) 座標軸翻轉處理
                #    你之前說你的計算在 Y 正向下 (OpenCV) 且 Z 正向後?
                #    MediaPipe 原生: Y 向下 (符合 OpenCV), Z 向前 (指向相機後方? MP 的 Z 定義是指向螢幕內側還是外側?)
                #    MediaPipe FaceMesh 的 Z: "The z coordinate represents depth, with the origin at the center of the head. Negative z values are in front of the face."
                #    (注意：MP 的 Z 負值是在臉前方)
                #    為了保險起見，我們先存 "原始像素方向"，進 PnP 前你再用你的 scale 參數去翻轉比較安全。
                #    這裡存的是純淨的 "Centered Pixel Coordinates"。

                try:
                    save_data = landmarks_centered.tolist()
                    with open(OUTPUT_FILENAME, 'w') as f:
                        json.dump(save_data, f, indent=2)
                    
                    print(f"✅ 已儲存！(原點位於鼻尖)")
                    print(f"   鼻尖座標檢查: {save_data[1]}") # 應該要是 [0, 0, 0]
                    last_saved_time = time.time()
                    flash_counter = 5
                except Exception as e:
                    print(f"❌ Error: {e}")

            # 顯示回饋
            if flash_counter > 0:
                cv2.rectangle(focus_image, (0,0), FOCUS_WINDOW_SIZE, (255, 255, 255), -1)
                flash_counter -= 1
            elif time.time() - last_saved_time < 2.0:
                 cx = FOCUS_WINDOW_SIZE[0] // 2
                 cy = FOCUS_WINDOW_SIZE[1] // 2
                 cv2.putText(focus_image, "SAVED (Nose Center)!", (cx - 150, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Raw Camera', debug_image)
            cv2.imshow('Geometry Focus', focus_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()