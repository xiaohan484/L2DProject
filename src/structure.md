# Project Structure Documentation

本文件旨在梳理 `src/` 目錄下的程式碼脈絡，供開發者快速查閱功能位於何處，以及如何調用，而無需閱讀完整原始碼。

## Execution Control
Please use `uv run ...` instead of `python ...` for future control.

## 目錄結構概覽

- **核心邏輯 (Core Logic)**
    - `mainArcade.py`: 程式入口，負責視窗管理、主迴圈、整合各模組。
    - `live2d.py`: 定義 Live2D 模型部件 (Live2DPart) 的層級結構與矩陣更新邏輯。
    - `deformer_system.py`: (NEW) Deformer 介面與實作 (Perspective, Expression)。
    - `parallax_deformer.py`: (Legacy) 處理非線性頂點變形 (Non-Linear Vertex Deformation) 以模擬頭部轉動的立體感 (Yaw/Pitch)。
    - `Const.py`: 全域常數設定。

- **渲染與網格 (Rendering & Mesh)**
    - `mesh_renderer.py`: 基於 Arcade 與 OpenGL 的網格渲染器 (GridMesh, SkinnedMesh)。
    - `create_mesh.py`: 圖像轉網格 (Triangulation) 及 物理模擬 (PBD) 算法。

- **動態回應 (Responses & Physics)**
    - `response.py`: 定義各部件如何回應偵測到的數據 (眨眼、張嘴、頭部旋轉)。
    - `ValueUtils.py`: 通用數學工具與物理計算類別 (SpringDamper, PendulumPhysics)。

- **追蹤輸入 (Tracking)**
    - `tracker.py`: 整合 MediaPipe Face Mesh，負責人臉特徵點捕捉與姿態計算。

---

## 詳細檔案說明

### 1. `mainArcade.py` (Main Entry)
程式的主要入口點。

#### Classes
- **`Live2DEngine(arcade.Window)`**
    - **用途**: 主應用程式視窗。
    - **`__init__(self, tracker)`**: 初始化視窗、追蹤器、物理系統、Live2D 根節點。
    - **`on_update(self, delta_time)`**: 更新迴圈。從 tracker 獲取數據 -> `update_pose` -> `root.update`。
    - **`on_draw(self)`**: 繪製迴圈。遍歷所有部件呼叫 `draw()`。
    - **`update_pose(self, face_info)`**: 平滑化頭部旋轉數據。

- **`TestMesh(arcade.Window)`**
    - **用途**: 用於測試單一網格生成與物理模擬的獨立場景。
    - **功能**: 生成網格 (`create_image_mesh`) -> 建立 `CustomMesh` -> 運行 `PBDCloth2D` 物理。

### 2. `live2d.py` (Model Hierarchy)
負責管理 Live2D 模型的部件層級與變形邏輯。

#### Global Variables
- **`func_table`**: 定義模型部件的層級關係與回應函數。
    - Format: `{"PartName": (Z_Depth, ResponseFunc, "ParentName", Optional[ConfigDict])}`
    - `ConfigDict` example: `{"type": CFG_PHYSICS, "stiffness": 0.2, "fixed_ratio": 0.1}`

#### Functions
- **`create_live2dpart(ctx)`** -> `(lives, root)`
    - 根據 `func_table` 自動建構完整的 Live2DPart 樹狀結構。
- **`load_local_position(...)`**: 計算子物件相對於父物件的初始偏移量。

#### Classes
- **`Live2DPart`**
    - **用途**: 代表模型的一個部件 (如頭髮、眼睛、身體)。
    - **屬性**: `x, y, angle, scale`, `local_matrix`, `world_matrix`, `children`.
    - **`update(self, data)`**: 更新自身的矩陣 (Local -> World)，並遞迴更新子節點。
        - 若配置了物理求解器 (`physics_solver`)，會在 World Space 進行物理模擬並更新 Mesh。
    - **`sync(self)`**: 將計算好的 World Matrix 同步給渲染層 (`views`).
    - **`draw(self)`**: 呼叫渲染層進行繪製。
    - **`NonLinearParallaxDeformer`**: (Defined in `deformer.py` but used here)
        - **用途**: 計算非線性頂點變形。
        - **`get_deformed_vertices(yaw, pitch)`**: 輸入旋轉角度，回傳變形後的頂點。
        - **`get_deformed_point(point, yaw, pitch)`**: 輸入單點座標，回傳變形後的座標 (用於子物件定位)。

### 2.1 `deformer_system.py` (Deformer Stack)
New modular system for deformation.

#### Classes
- **`BaseDeformer`**: Abstract interface.
    - **`transform(vertices, params)`**: Main deformation logic.
    - **`transform_and_scale(point, params)`**: Helper for child attachment (returns scale).
- **`PerspectiveDeformer`**: Wraps `NonLinearParallaxDeformer` from `parallax_deformer.py`.
- **`ExpressionDeformer`**: Placeholder for shape deformations.

### 2.2 `parallax_deformer.py` (Legacy Logic)
獨立的變形算法模組 (Backend for PerspectiveDeformer).

#### Classes
- **`NonLinearParallaxDeformer`**
    - **`__init__(self, pts, radius_scale)`**: 初始化變形器，計算網格半徑與權重 (Weights)。
    - **原理**:
        - **Bulge**: 中心點移動量大，邊緣移動量小 (Parallax)。
        - **Squash/Stretch**: 根據旋轉方向壓縮遠端、拉伸近端 (Perspective)。
    - **權重公式**: $1 - (r/R)^3$。

### 3. `mesh_renderer.py` (Rendering)
處理底層 OpenGL 繪圖。

#### Classes
- **`GridMesh`**
    - **用途**: 基礎網格渲染器，支援 N x M 的網格變形。
    - **`setup_mesh_data(...)`**: 生成頂點 (Vertices) 與索引 (Indices)。
    - **`apply_bend(self, bend_amount)`**: 簡單的彎曲變形算法 (針對長條狀物體)。
    - **`update_buffer(self)`**: 將 CPU 的頂點資料上傳至 GPU。
    - **`draw(self, world_matrix)`**: 設定 Shader Uniforms 並執行 Draw Call。

- **`SkinnedMesh(GridMesh)`**
    - **用途**: 支援骨骼綁定 (Linear Blend Skinning)。
    - **`auto_smooth_bind(self)`**: 自動計算頂點權重 (基於距離)。
    - **`update_skinning(self)`**: 根據骨骼位置更新頂點座標 (CPU Skinning)。

- **`CustomMesh(GridMesh)`**
    - **用途**: 支援任意三角形網格 (非規則 Grid)，通常配合 `create_mesh.py` 使用。

### 4. `create_mesh.py` (Mesh & Physics)
負責從圖片生成網格以及物理模擬。

#### Functions
- **`create_image_mesh(image_path, debug)`** -> `(pts, tris)`
    - 使用 OpenCV 偵測圖片輪廓，並使用 `triangle` 庫進行三角剖分 (Delaunay Triangulation)。

#### Classes
- **`PBDCloth2D`**
    - **用途**: 2D 位置基礎動力學 (Position Based Dynamics) 模擬器 (用於頭髮飄動)。
    - **`step(self, iterations)`**: 執行一次物理模擬步進 (重力 -> 預測位置 -> 距離約束修正 -> 更新速度)。
    - **`_solve_distance_constraints(...)`**: 維持邊長不變的約束求解。

### 5. `tracker.py` (Face Tracking)
負責與 MediaPipe 互動。

#### Classes
- **`FaceTracker`**
    - **用途**: 封裝 MediaPipe Face Mesh。
    - **`process(self)`**: 讀取 Webcam 影像並進行偵測。
    - **`get_head_pose(self, ...)`**: 使用 `solvePnP` 計算頭部旋轉 (Yaw, Pitch, Roll)。
    - **`get_iris_pos(self)`**: 從 Blendshapes 計算眼球位置。
    - **`get_eye_blink_ratio(self)`**: 計算眨眼程度。

- **`AsyncFaceTracker`**
    - **用途**: 在獨立 Thread 運行 `FaceTracker`，避免卡住主視窗。
    - **`_update_loop(self)`**: 持續更新追蹤數據並透過 `pubsub` 發送。

- **`FakeTracker`**
    - **用途**: 產生假數據，用於沒有 Webcam 時的測試。

### 6. `response.py` (Response Logic)
定義模型如何根據輸入數據產生動作。

#### Functions
- **`body_response(live, data)`**: 呼吸效果 (Scale 變化)。
- **`face_response(live, data)`**: 頭部旋轉 (Z軸 Roll)。
- **`pupils_response(dir, live, data)`**: 眼球移動 (限制在橢圓範圍內)。
- **`lid_response(...)`**: 眼皮開闔 (連動睫毛)。
- **`mouth_response(...)`**: 嘴型切換 (Texture Swap)。
- **`Partial Functions`**: 如 `pupils_response_l` 預先綁定左右參數。

### 7. `ValueUtils.py` (Utilities)
#### Functions
- **`map_range(value, in_min, in_max, out_min, out_max)`**: 數值映射。

#### Classes
- **`SpringDamper`**: 簡易彈簧阻尼系統 (用於物理參數計算)。
- **`PendulumPhysics`**: 單擺物理系統 (用於頭髮擺動模擬)。

### 8. `Const.py` (Configuration)
包含 `MODEL_PATH`, `CANVAS_HEIGHT`, `SCREEN_WIDTH` 等全域設定。
以及載入 `model_data.json`。
