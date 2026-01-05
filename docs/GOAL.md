📋 Python VTuber Engine - Project Handover Protocol
【專案核心定義】

目標： 開發一個 自製的 VTuber 渲染引擎 (非使用 VTube Studio/Unity)。

渲染庫： Python + arcade (或未來的 OpenGL/ModernGL)。

核心邏輯： MediaPipe (偵測) -> OpenCV/NumPy (數學計算) -> arcade (繪圖/圖層操作)。

當前階段： Level 2 - 2.5D Sprite Animation (圖層視差 + 圖庫切換)。

未來階段： Level 3 - Mesh Deformation (自製網格變形引擎)。

【已完成技術模組 (The Assets)】

眼部系統： 虹膜追蹤 (Gaze) 與 眨眼偵測 (EAR)，數值已就緒。

高精度姿態 (The PnP Engine)：

已完成「個人化臉模」校正 (鼻尖原點)。

已透過「逆向剛體變換」驗證 PnP 的旋轉 (rvec) 是物理正確的。

價值： 這組 rvec 將被用來驅動圖層視差，而不僅僅是轉頭。

【當前開發規劃 (Roadmap)】

Step 1: 圖層視差 (Parallax) - [進行中]

原理： 利用 PnP 算出的 rvec (Yaw/Pitch)，計算「後髮」、「臉」、「前髮」的相對位移 (dx, dy)。

目的： 創造 2.5D 的立體感，這是 Level 2 引擎的核心靈魂。

Step 2: 嘴部系統 (Mouth) - [下一步]

策略： 暫緩 Mesh 變形。採用 「圖庫切換法 (Sprite Swapping)」。

實作： 準備閉嘴、微張、全開等 Sprite，根據開合數值切換。

需求： 仍需實作 calibrate_mouth 來計算 0.0~1.0 的開合參數。

Step 3: 自製 Mesh 引擎 (Level 3) - [遠期目標]

策略： 當 Sprite 效果到達極限後，再切入 OpenGL Shader 與頂點變形。目前先以完成「能動的角色」為優先。