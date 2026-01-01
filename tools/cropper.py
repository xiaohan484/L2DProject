import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import filedialog
from jsonUtils import *

# 設定
INPUT_DIR = "assets/sample_model/raw"       # 放原本的全畫布 PNG
OUTPUT_DIR = "assets/sample_model/processed" # 這裡會吐出裁切後的 PNG 和 JSON
CONFIG_FILE = "model_data.json" # 最終存檔

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AssetTool:
    def __init__(self):
        self.data = {}
        self.current_img_name = ""
        self.original_img = None
        self.cropped_img = None
        self.offset_x = 0  # 裁切後的圖在原圖的 x 偏移
        self.offset_y = 0  # 裁切後的圖在原圖的 y 偏移
        self.anchor = (0.5, 0.5) # 預設錨點 (正規化 0~1)

    def select_files(self):
        root = tk.Tk()
        root.withdraw() # 隱藏主視窗
        file_paths = filedialog.askopenfilenames(
            title="選擇要處理的 PNG (全畫布)",
            filetypes=[("PNG Images", "*.png")],
            initialdir=INPUT_DIR
        )
        return file_paths

    def auto_crop(self, image):
        """
        演算法：找 Alpha 通道不為 0 的 Bounding Box
        """
        # 取得 Alpha channel
        b, g, r, a = cv2.split(image)
        
        # 找所有不透明的點
        coords = cv2.findNonZero(a)
        if coords is None:
            print(f"⚠️ {self.current_img_name} 是一張全透明的圖！跳過。")
            return None, 0, 0
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # 裁切
        cropped = image[y:y+h, x:x+w]
        return cropped, x, y

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 取得圖片長寬
            h, w = self.cropped_img.shape[:2]
            
            # 計算正規化錨點 (0.0 ~ 1.0)
            anchor_x = x / w
            anchor_y = y / h
            self.anchor = (anchor_x, anchor_y)
            print(f"📍 錨點設定: ({x}, {y}) -> {self.anchor}")
            
            # 畫個圈圈顯示目前選的點
            display_img = self.cropped_img.copy()
            # 畫十字
            cv2.line(display_img, (x-10, y), (x+10, y), (0, 0, 255), 1)
            cv2.line(display_img, (x, y-10), (x, y+10), (0, 0, 255), 1)
            cv2.imshow("Set Anchor (Press SPACE to confirm)", display_img)

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]
        self.current_img_name = name_no_ext
        
        # 1. 讀取圖片 (保留 Alpha)
        # OpenCV 預設讀取路徑不支援中文，這裡用 numpy workaround
        img_array = np.fromfile(filepath, np.uint8)
        self.original_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        if self.original_img is None:
            print(f"❌ 無法讀取 {filename}")
            return

        # 2. 自動裁切
        print(f"✂️ 正在裁切 {filename}...")
        self.cropped_img, self.offset_x, self.offset_y = self.auto_crop(self.original_img)
        
        if self.cropped_img is None: return

        # 3. 互動設定 Anchor
        cv2.imshow("Set Anchor (Press SPACE to confirm)", self.cropped_img)
        cv2.setMouseCallback("Set Anchor (Press SPACE to confirm)", self.on_mouse_click)
        
        print(f"👉 請在視窗中點擊 {name_no_ext} 的旋轉/縮放中心 (例如瞳孔中心、脖子根部)。按空白鍵確認。")
        
        while True:
            key = cv2.waitKey(0)
            if key == 32: # Space 鍵
                break
        
        cv2.destroyAllWindows()

        # 4. 存檔 (裁切後的 PNG)
        output_png_path = os.path.join(OUTPUT_DIR, filename)
        # 同樣 workaround 存檔中文/特殊路徑問題
        is_success, im_buf = cv2.imencode(".png", self.cropped_img)
        if is_success:
            im_buf.tofile(output_png_path)
            print(f"💾 已儲存圖片: {output_png_path}")

        # 5. 紀錄資料
        # 我們存兩個座標：
        # - center_x/y: 這張小圖的中心點，對應到原本大螢幕的哪個絕對座標？(給 Arcade 用)
        # - anchor_x/y: 這張小圖的旋轉中心在哪？(正規化 0~1)
        
        h, w = self.cropped_img.shape[:2]
        
        # 算出這張小圖的"幾何中心"在原本大畫布的哪裡
        # 原本大畫布座標 = 裁切偏移(offset) + 小圖的一半(w/2)
        global_center_x = self.offset_x + (w / 2)
        
        # OpenCV 的 y 是由上往下算，但 Arcade 也是，只是 Arcade 顯示時可能要在視窗翻轉
        # 我們這裡先記錄 "Raw Pixel Coordinate" (原圖座標)，程式裡再轉換
        global_center_y = self.offset_y + (h / 2)

        self.data[name_no_ext] = {
            "filename": filename,
            "original_width": w,
            "original_height": h,
            "global_center_x": int(global_center_x),
            "global_center_y": int(global_center_y), # 這是大圖上的絕對座標
            "anchor_x": round(self.anchor[0], 4),    # 0.5 = 中心
            "anchor_y": round(self.anchor[1], 4)     # 0.5 = 中心
        }
        print("------------------------------------------------")

    def save_json(self):
        json_path = os.path.join(OUTPUT_DIR, CONFIG_FILE)
        setJsonPath(json_path)
        addDataDict(self.data)

if __name__ == "__main__":
    tool = AssetTool()
    files = tool.select_files()
    if files:
        for f in files:
            tool.process_file(f)
        tool.save_json()
    else:
        print("未選擇檔案")