import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SkeletonChainFABRIK:
    def __init__(self, n_bones=10, bone_length=1.0):
        self.n = n_bones
        self.L = bone_length

        # 1. 關節位置 (Joint Positions)
        # P[i] 是第 i 根骨頭的 "起點"
        # P[n] 是最後一根骨頭的 "終點"
        self.points = np.zeros((n_bones + 1, 2))
        for i in range(n_bones + 1):
            self.points[i] = [0, -i * bone_length]

        # 2. 骨架矩陣 (Transforms) - 我們要計算的目標
        # 儲存每根骨頭的 [Ox, Oy, Xx, Xy, Yx, Yy] (原點 + 基底向量)
        self.transforms = np.zeros((n_bones, 6))

    def update(self, target_head_pos, dt):
        # --- A. 應用外力 (簡單模擬重力/風力) ---
        # 在 FABRIK 中，外力通常先作用在點上，讓點 "暫時" 跑掉
        # 然後再透過 FABRIK 強制拉回來

        # 模擬重力 + 風力
        gravity = np.array([0, -9.8]) * dt * dt
        wind_noise = np.sin(target_head_pos[0] * 0.5 + self.points[:, 1]) * 0.05

        for i in range(1, self.n + 1):  # 頭部(0)不動物理
            self.points[i] += gravity
            self.points[i, 0] += wind_noise[i] + 0.02  # 風向右吹

        # --- B. FABRIK 核心算法 (約束求解) ---
        # 迭代幾次以確保骨架長度正確
        for _ in range(3):
            self._solve_fabrik(target_head_pos)

        # --- C. 計算骨架矩陣 (Skeleton Calculation) ---
        self._compute_transforms()

    def _solve_fabrik(self, head_pos):
        # 1. Backward Reach (從頭開始，固定住頭)
        self.points[0] = head_pos
        for i in range(self.n):
            p_curr = self.points[i]
            p_next = self.points[i + 1]

            vec = p_next - p_curr
            dist = np.linalg.norm(vec)
            if dist == 0:
                dist = 0.001

            # 拉回正確長度
            self.points[i + 1] = p_curr + (vec / dist) * self.L

        # 2. Forward Reach (從尾巴開始，如果有目標點的話)
        # 如果是自由擺動的繩子，Forward Reach 其實可以省略，
        # 或者是用來限制 "尾巴不能穿過地板" 等約束。
        # 這裡為了單純模擬 "被頭拖著走"，Backward Reach 其實就夠了，
        # 但標準 FABRIK 會包含這步。
        pass

    def _compute_transforms(self):
        """
        這就是你要的：計算每一節骨架的 Local Coordinates
        """
        for i in range(self.n):
            # 骨頭起點
            origin = self.points[i]
            # 骨頭終點
            end = self.points[i + 1]

            # Local X (Bone Axis): 指向下一節
            x_axis = end - origin
            norm = np.linalg.norm(x_axis)
            if norm > 0:
                x_axis /= norm

            # Local Y (Normal): 垂直於 X
            y_axis = np.array([-x_axis[1], x_axis[0]])

            # 存入資料結構 [Ox, Oy, Xx, Xy, Yx, Yy]
            self.transforms[i] = [
                origin[0],
                origin[1],
                x_axis[0],
                x_axis[1],
                y_axis[0],
                y_axis[1],
            ]


# --- 視覺化骨架座標軸 ---
sim = SkeletonChainFABRIK(n_bones=8, bone_length=1.5)

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-15, 2)
ax.set_aspect("equal")
ax.grid(True)

# 畫骨架連線 (黑色)
(line,) = ax.plot([], [], "k-", lw=2)
# 畫每一節的 Local X 軸 (紅色)
quiver_x = ax.quiver([], [], [], [], color="r", scale=10, width=0.005, headwidth=3)
# 畫每一節的 Local Y 軸 (綠色)
quiver_y = ax.quiver([], [], [], [], color="g", scale=10, width=0.005, headwidth=3)


def animate(frame):
    dt = 0.033
    t = frame * dt

    # 讓頭部左右移動
    head_pos = np.array([np.sin(t * 3) * 4, 0])

    sim.update(head_pos, dt)

    # 更新連線
    line.set_data(sim.points[:, 0], sim.points[:, 1])

    # 更新座標軸向量 (Quiver)
    origins_x = sim.transforms[:, 0]
    origins_y = sim.transforms[:, 1]
    vecs_xx = sim.transforms[:, 2]
    vecs_xy = sim.transforms[:, 3]
    vecs_yx = sim.transforms[:, 4]
    vecs_yy = sim.transforms[:, 5]

    quiver_x.set_offsets(np.c_[origins_x, origins_y])
    quiver_x.set_UVC(vecs_xx, vecs_xy)

    quiver_y.set_offsets(np.c_[origins_x, origins_y])
    quiver_y.set_UVC(vecs_yx, vecs_yy)

    return line, quiver_x, quiver_y


ani = FuncAnimation(fig, animate, frames=200, interval=33, blit=False)
plt.show()
