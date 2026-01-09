# 📋 Python VTuber Engine - Project Handover Protocol

## level 3 開發路徑圖 (Roadmap)
我們將這個階段拆解為四個里程碑：

### Step 1: The Grid (網格生成) 🛠️

目標：捨棄 arcade.Sprite。學會如何用 Python 生成一組 N x M 的頂點資料 (Vertices) 與 紋理座標 (UVs)，並讓 arcade 把它畫出來。

產出：能在螢幕上看到一張被切分成很多格子的圖片（Wireframe 模式或正常顯示）。

### Step 2: The Wiggle (基礎變形) 🌊

目標：證明我們能「動」它。

實作：對頂點套用簡單的 Sin/Cos 波形函數。

產出：看到頭髮像旗幟一樣飄動（雖然很不自然，但證明引擎活了）。

### Step 3: The Control (控制點系統) 🎮

目標：實作 FFD (Free-Form Deformation) 的簡化版。

實作：我們不能手調幾百個點。我們要定義 4-9 個「控制點 (Control Points)」，用插值算法 (Interpolation) 來驅動整個網格。

產出：拉動一個點，整條頭髮平滑彎曲。

### Step 4: Integration (物理與參數掛載) 🔗

目標：將之前的 PnP (頭轉) 與 物理 (彈簧) 數值，掛載到 Step 3 的控制點上。

產出：真正的 Live2D 效果——頭髮隨頭部轉動而柔順甩動。