# L2DProject

> **🚧 Project Status: Active Development / WIP**
>
> This project is currently under construction. I am porting my local experiments into this repository.
> Expect breaking changes and refactoring in the coming days.

## 📖 Overview
This project is a technical exploration into building a real-time VTuber system using pure Python.

Unlike standard solutions that rely on Unity or heavy game engines, my goal is to push the limits of Python's ecosystem (OpenCV + Live2D bindings) to see if we can achieve smooth, low-latency character animation with lightweight dependencies.

(本專案是一個技術探索，旨在測試使用純 Python 建構即時 VTuber 系統的可能性。不同於依賴 Unity 等大型引擎的正規解法，我想挑戰 Python 生態系的極限，測試是否能在輕量級依賴下實現流暢、低延遲的角色動畫。)

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (MediaPipe backbone)
* **Signal Processing:** One Euro Filter (for jitter reduction)
* **Rendering:** [Arcade / ...]

![Demo](assets/demo_preview5.gif)

WIP Prototype: Real-time eye blinking and eye tracking driven by live facial landmark data. (Note: Actual face input video is not shown for privacy reasons.)

## 🗺️ Roadmap

<details>
<summary><strong>Phase 1: Let it moves </strong></summary>

- Basic Face Detection (MediaPipe)
    - [x] Async Tracking
    - [x] Head's pose estimation and rotation

- 2D character movement
    - [x] Gaze (One Euro Filter)
    - [x] Breathing (Simple sin solution)
    - [x] Blinking (One Euro Filter)

</details>

<details>
<summary><strong>Phase 2: Parallax(Current Focus) </strong></summary>

- Face Detection (MediaPipe)
    - [x] Head Pose Estimation(Yaw,Pitch)
    - [ ] Refine Existing Solution more precisely(not x,y only)
        - [ ] Pupils
        - [ ] Blinking
        - [ ] Mouth

- 2D character movement
    - [x] Head rotation
    - [x] Mouth

</details>

<details>
<summary><strong>Phase 3: Mesh </strong></summary>

- Face Detection (MediaPipe)
    - [ ] LandMark Remapping(partial)

- 2D character movement
    - [ ] Mouth (Mesh)
    - [ ] Physics

</details>

## ⚠️ Disclaimer & Credits
* **Assets:** This project uses Live2D assets for demonstration purposes. 
    * Character Art: AI-generated concepts (Nano Banana Pro), processed for Live2D rigging by me.
    * **Note:** No proprietary model files (`.moc3`) are included in this repo to respect licensing.
* **License:** MIT License (See `LICENSE` file).