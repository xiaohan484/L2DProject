# 技術實作細節與開發心得 (Technical Implementation Notes)

## 1 頭部姿態估計 (Pose Estimation)
#### 1.1 Mathematics model of PnP Problem
##### 1.1.1 Perspective Projection Model

相機拍攝物體的過程，在數學上描述為將相機座標系中的 3D 點 $\mathbf{P}_c = [x, y, z]^T$ 投影到 2D 圖像平面 $\mathbf{p} = [u, v]^T$ 的過程。

根據針孔相機模型 (Pinhole Camera Model)，其非線性關係為：

$$ u = \frac{f_xx}{z} +c_x$$
$$ v = \frac{f_yy}{z} + c_y$$

為了便於線性代數運算，我們引入 **齊次座標 (Homogeneous Coordinates)**，將其寫成矩陣形式：

$$ s\begin{bmatrix} u \\ v \\1 \end{bmatrix} = K \begin{bmatrix}x \\ y \\ z\end{bmatrix} $$

其中：
* $s$ 為尺度因子 (Scale Factor)，在此模型中 $s = z$ (即點的深度)。
* $\mathbf{K}$ 為 **內參矩陣 (Intrinsic Matrix)**，描述相機的光學特性：
$$
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

> **Note:**
> 齊次座標的特性意味著向量 $[u, v, 1]^T$ 與 $[su, sv, s]^T$ 代表同一個 2D 點。因此，從矩陣形式回到歐幾里得座標 (Euclidean Coordinates) 時，必須執行 **透視除法 (Perspective Division)**，即將向量除以最後一個分量 $z$。

#### 1.1.2 Sensitivity Analysis (敏感度分析)

對投影方程進行偏微分，我們可以觀察像素變化對 3D 位移的敏感度，這解釋了為何某些姿態下 PnP 解算會不穩定。

**1. 對平移 $x, y$ 的敏感度：**
$$
\frac{\partial u}{\partial x} = \frac{f_x}{z}, \quad \frac{\partial v}{\partial y} = \frac{f_y}{z}
$$
**物理意義**：相機距離目標點越遠 ($z$ 越大)，$\frac{1}{z}$ 越小。這意味著遠處物體的 $x, y$ 移動在畫面上產生的像素位移量 ($u, v$) 較小，導致訊號雜訊比 (SNR) 降低。

**2. 對深度 $z$ 的敏感度 (關鍵)：**
$$
\frac{\partial u}{\partial z} = -f_x \frac{x}{z^2}, \quad \frac{\partial v}{\partial z} = -f_y \frac{y}{z^2}
$$
**物理意義**：
* 深度變化的影響與 $z^2$ 成反比，隨距離衰減極快。
* $x,y$越大(如越接近畫面邊緣)，更容易觀測到$z$的運動，反之也成立。

綜合起來的話，就是拍攝的時候，想要觀測的物體不要直接正對中心，稍有偏移反而更好。(但對於精準度又要求的時候，距離相機邊緣畸變越嚴重，須注意校正以及不要真的移動到邊緣)

#### 1.1.3 Coordinate Transformation & Extrinsics (座標變換與外參)

在電腦視覺中，我們需要處理不同的座標參考系。對於任意物體 $O$，我們定義其自身的座標系為 **模型座標系 (Model Coordinate System)**，物體上任意特徵點 $i$ 的座標表示為 $\mathbf{P}_m = [X_m, Y_m, Z_m]^T$。

我們的目標是將這些點轉換到 **相機座標系 (Camera Coordinate System)** $\mathbf{P}_c = [X_c, Y_c, Z_c]^T$。這兩者之間的關係由剛體變換 (Rigid Body Transformation) 描述：

$$
\mathbf{P}_c = \mathbf{R} \mathbf{P}_m + \mathbf{t}
$$

或者寫成矩陣形式：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{R} \begin{bmatrix} X_m \\ Y_m \\ Z_m \end{bmatrix} + \mathbf{t}
$$

其中：
* $\mathbf{R}$：$3 \times 3$ 的 **旋轉矩陣 (Rotation Matrix)**，屬於特殊正交群 $SO(3)$。
* $\mathbf{t}$：$3 \times 1$ 的 **平移向量 (Translation Vector)**。
* 這兩者構成了相機的 **外部參數 (Extrinsic Parameters)**，共包含 6 個自由度 (3 個旋轉角度 + 3 個位移分量)。

**結合投影模型：**

將上述變換代入 1.1.1 的投影公式，我們得到完整的 **PnP 投影方程式**：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \left( \mathbf{R} \begin{bmatrix} X_m \\ Y_m \\ Z_m \end{bmatrix} + \mathbf{t} \right)
$$

為了簡化書寫，常將旋轉與平移合併為一個 $3 \times 4$ 的外參矩陣 $[ \mathbf{R} | \mathbf{t} ]$，並將模型點寫成齊次座標 $[X_m, Y_m, Z_m, 1]^T$：

$$
s \mathbf{p} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \mathbf{P}_{homo}
$$

**求解條件 (PnP Constraints)：**

* **未知數**：我們已知 3D 模型點 $\mathbf{P}_m$ 和 2D 觀測點 $\mathbf{p}$，已知內參 $\mathbf{K}$，要求解的是 $\mathbf{R}$ 和 $\mathbf{t}$。
* **最少點數要求**：
    * **3 點 (P3P)**：理論上的最小需求（因為每個點提供 2 個方程，3 點提供 6 個方程對應 6 個未知數）。但 P3P 會產生多達 4 組解 (Multimodal solutions)，需要第 4 點來消歧義。
    * **4 點以上 (PnP)**：實務上，由於觀測雜訊 (Pixel Noise) 的存在，我們會使用 $N \ge 4$ 個點，透過最小平方法 (Least Squares) 來最小化重投影誤差。

**求解條件 (PnP Constraints)：**

* **自由度分析**：外部參數 $[\mathbf{R}|\mathbf{t}]$ 包含 6 個自由度 (3 旋轉 + 3 平移)。
* **最小點數需求**：
    * 理論上，3 組對應點 (P3P) 可提供 6 個約束方程，足以求解，但會產生多組數學解 (Multiple Solutions)。
    * 工程實務上，通常需要 $N \ge 4$ 點來消除歧義並降低觀測雜訊的影響。
* **幾何分佈限制 (Geometric Configuration)**：
    雖然數學上允許特徵點共面 (Co-planar)，但在求解 PnP 時，若所有特徵點皆位於同一平面（即 $Z_m \approx \text{const}$），會導致投影矩陣的奇異值分佈惡化，使問題變為 **病態 (Ill-conditioned)**。這意味著對輸入雜訊的敏感度極高，極微小的像素誤差可能導致算出的位姿產生巨大偏差。因此，特徵點在 3D 空間中的立體分佈 (Spatial Diversity) 是影響解算穩定性的關鍵因素。

### 1.2 The PnP Optimization Problem (PnP 最佳化問題)

將幾何投影結合實際觀測，我們將 PnP 定義為一個 **非線性最小平法問題 (Non-linear Least Squares Problem)**。

假設我們有 $N$ 組對應點 ($N \ge 4$)：
1.  **3D Model Points**: 已知物體座標系下的特徵點 $\mathbf{P}_m^{(i)} = [X_m^{(i)}, Y_m^{(i)}, Z_m^{(i)}]^T$。
2.  **2D Observed Points**: 從相機畫面 (MediaPipe) 觀測到的像素座標 $\mathbf{p}_{obs}^{(i)} = [u_{obs}^{(i)}, v_{obs}^{(i)}]^T$。

由於感測器雜訊 (Sensor Noise)、特徵點偵測誤差以及數值精度限制，觀測點 $\mathbf{p}_{obs}$ 與理想投影點 $\mathbf{p}_{proj}$ 之間必然存在誤差。

我們的目標是尋找最佳的旋轉矩陣 $\mathbf{R}$ 與平移向量 $\mathbf{t}$，使得所有特徵點的 **重投影誤差 (Reprojection Error)** 總和最小化。數學上表示為：

$$
\{ \mathbf{R}^*, \mathbf{t}^* \} = \operatorname*{argmin}_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \| \mathbf{p}_{obs}^{(i)} - \pi(\mathbf{P}_m^{(i)}, \mathbf{K}, \mathbf{R}, \mathbf{t}) \|^2
$$

其中：
* $\pi(\cdot)$ 代表 1.1 節中定義的透視投影函數 (Perspective Projection Function)，將 3D 點映射至 2D 平面。
* $\| \cdot \|^2$ 代表歐幾里得距離的平方 (Squared Euclidean Norm)。

**誤差函數展開 (Residual Function):**

具體而言，我們希望最小化的誤差函數 $E$ 為：

$$
E(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^{N} \left[ \left( u_{obs}^{(i)} - \left( f_x \frac{X_c^{(i)}}{Z_c^{(i)}} + c_x \right) \right)^2 + \left( v_{obs}^{(i)} - \left( f_y \frac{Y_c^{(i)}}{Z_c^{(i)}} + c_y \right) \right)^2 \right]
$$

其中 $X_c, Y_c, Z_c$ 是由 $\mathbf{R}, \mathbf{t}$ 變換後的相機座標。

> **Algorithm Note:**
> 由於投影函數 $\pi$ 包含除法運算 ($\frac{1}{Z_c}$) 以及旋轉矩陣 $\mathbf{R}$ 的三角函數，這是一個高度 **非線性 (Non-linear)** 的最佳化問題。
> 因此，無法使用簡單的線性代數求解，通常需要採用 **迭代法 (Iterative Methods)**，如 **Levenberg-Marquardt (LM) Algorithm**。這也解釋了為什麼在 OpenCV 中使用 `SOLVEPNP_ITERATIVE` 時，需要提供一個良好的初始猜測值 (Initial Guess)，否則容易陷入局部極小值 (Local Minima)。

### 1.2.1 Jacobian Sensitivity Analysis (敏感度分析)

為了理解 PnP 解算的穩定性，我們需要分析觀測值 $(u,v)$ 對於狀態變數 $(R, t)$ 的 **Jacobian Matrix (偏微分矩陣)**。以下以 $u$ 分量為例進行推導。

#### 1. Translation Sensitivity (平移敏感度)

$$
\frac{\partial u}{\partial t_x} = \frac{\partial}{\partial t_x} \left( f_x \frac{x + t_x}{z} + c_x \right) = \frac{f_x}{z}
$$

**觀察**：敏感度與深度 $z$ 成反比。物體越遠，平移造成的像素變化越不明顯，信噪比 (SNR) 越低。

#### 2. Rotation Sensitivity (旋轉敏感度)

針對旋轉矩陣 $\mathbf{R}$，我們採用 Euler Angles ($Z-Y-X$ 順序) 進行分解：
$$\mathbf{R} = \mathbf{R}_z(\theta_z) \mathbf{R}_y(\theta_y) \mathbf{R}_x(\theta_x)$$

展開後的矩陣雖複雜，但在追蹤過程中，我們可定義 **局部切空間 (Local Tangent Space)**，假設當前姿態的微小變化量 $\theta \approx 0$。利用小角度近似 ($\cos\theta \approx 1, \sin\theta \approx \theta$)，旋轉矩陣線性化為：

$$
\mathbf{R} \approx \begin{bmatrix}
1 & -\theta_z & \theta_y \\
\theta_z & 1 & -\theta_x \\
-\theta_y & \theta_x & 1
\end{bmatrix}
$$

**針對 Yaw 軸 ($\theta_y$) 的分析：**

考慮物體繞自身中心旋轉 (Object-centric Rotation)，其座標變化率為：
$$\frac{\partial x}{\partial \theta_y} = z_o, \quad \frac{\partial z}{\partial \theta_y} = -x_o$$
(其中 $x_o, z_o$ 為特徵點在模型座標系中的位置)

代入投影微分公式：

$$
\frac{\partial u}{\partial \theta_y} = \frac{f_x}{z^2} \left( z \frac{\partial x}{\partial \theta_y} - x \frac{\partial z}{\partial \theta_y} \right)
$$

得到最終敏感度公式：

$$
\frac{\partial u}{\partial \theta_y} = \frac{f_x}{z^2} ( \underbrace{z \cdot z_o}_{\text{Depth Term}} + \underbrace{x \cdot x_o}_{\text{Perspective Term}} )
$$

#### 3. Critical Analysis (關鍵分析)

這個公式揭露了正臉解算不穩定的數學成因：

1.  **平面退化 (Planar Degeneracy)**：
    第一項 $z \cdot z_o$ 依賴於特徵點的自身深度 $z_o$。若特徵點集中於面部平面 (如眼睛、嘴巴)，則 $z_o \approx 0$，導致此項訊息遺失。
2.  **中心退化 (Center Alignment)**：
    第二項 $x \cdot x_o$ 依賴於物體在畫面中的位置 $x$。當使用者位於畫面正中央 ($x \approx 0$) 時，此項亦趨近於零。

**Summary:**
* 當 **$z_o \approx 0$ (臉平)** 且 **$x \approx 0$ (居中)** 時，$\frac{\partial u}{\partial \theta_y} \approx 0$。此時 Jacobian 矩陣接近奇異 (Singular)，任何像素雜訊都會被錯誤放大為巨大的旋轉或平移跳動。
* **結論**：挑選特徵點時，必須盡可能最大化 $z_o$ 的變異數（例如引入耳朵、眉骨等深度點），以確保即使在 $x \approx 0$ 時，第一項 $z \cdot z_o$ 仍能提供足夠的梯度訊息。

#### 1.2 Choice of the Face Landmark

#### 1.2.1 Choice of the LandMark
儘量選擇 附著骨骼的特徵點，避開可張開的下巴或是可動的肌肉。
由於深度的需求，和其他特徵點深度差最大的鼻子也挑幾個特徵點。

#### 1.2.2 Output coordinate of MediaPipe
注意MediaPipe 的輸出 是x,y,z，為正規化座標系。這是因為[0,1]對深度學習的計算相對友好，以及 獨立於解析度的性質。

其中 x,y 的數值 要乘以 影像的長img_w,寬img_h。
z的數值要乘以img_w，但由於其數值沒有考慮到透射投影，如果對精度有較大的需求，不建議使用。

#### 1.2.3 Why not use transformation of MediaPipe output
MediaPipe output 是基於標準臉型 轉換至 偵測特徵點的變化矩陣，包含了Scaling的元素，不一貼合臉型，如果目標是取得Pose 精度可能有疑慮。
如果目標是輸出貼圖到自己臉上可能就蠻合適的。

#### 1.2.4 Face LandMark Registering
如果要註冊自己的臉模，可拍攝自己多張照片，取得每一個特徵點。
加入PnP 的優化。(這部分應該類似SLAM技術)

#### 1.3 Solver & Coordinate System
#### 1.4 Not Implement Now But it can be better

## 2. Time Consistent(Challenges & Solutions)
### 2.1 One Euro Filter vs Kalman Filter
### 2.2 Pupils and Blinking
### 2.3 Filter in Pose Estimation

---
*Last Updated: 2026-01-03*