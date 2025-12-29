import json
# --- 設定區 ---
# 請設為你原本 Krita 畫布的高度 (這很重要！用來做 Y 軸翻轉)
# 如果你的原圖是 1216x2236，這裡就填 2236
CANVAS_HEIGHT = 2236 
GLOBAL_SCALE = 1.5

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "My VTuber v0.2"
MODEL_DATA = {}
try:
    with open("assets/processed/model_data.json", "r", encoding="utf-8") as f:
        MODEL_DATA = json.load(f)
except FileNotFoundError:
    print("❌ 找不到 model_data.json，請先執行 tools/cropper.py")