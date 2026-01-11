import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ShakingRopeSimulation:
    def __init__(self, n_segments=40, total_length=15.0, start_pos=(0, 0)):
        self.n = n_segments
        self.segment_len = total_length / n_segments

        # 初始化：繩子垂直垂下
        self.pos = np.zeros((n_segments, 2))
        for i in range(n_segments):
            self.pos[i] = [start_pos[0], start_pos[1] - i * self.segment_len]

        self.prev_pos = self.pos.copy()

        # 物理參數
        self.gravity = np.array([0, -9.8])
        # 空氣阻力：設高一點 (0.9 ~ 0.95) 可以看到波動傳遞得更遠
        # 設低一點 (0.5) 繩子會很像在水裡甩動
        self.friction = 0.98

    def update(self, dt, time_elapsed):
        # --- 1. 更新 "手" (固定端) 的位置 ---
        # 這裡我們模擬左右移動：正弦波運動
        # Amplitude (振幅): 左右移動的寬度
        # Frequency (頻率): 甩動的快慢
        amplitude = 3.0
        frequency = 5.0

        head_x = np.sin(time_elapsed * frequency) * amplitude
        head_y = 0  # 高度維持不變

        # 強制設定第 0 點的位置 (Kinematic Control)
        # 注意：我們不需要手動計算它的速度，約束系統會自動把這個移動傳遞給下一節
        current_head_pos = np.array([head_x, head_y])
        self.pos[0] = current_head_pos

        # --- 2. 物理積分 (Verlet Integration) ---
        # 計算除了第 0 點以外的所有點

        # 移除風力，只保留重力
        # 如果你想模擬 "平放在桌面上甩動"，可以把 gravity 設為 [0, 0]
        total_acc = self.gravity

        velocity = self.pos - self.prev_pos
        last_pos = self.pos.copy()

        # Verlet 更新
        self.pos[1:] = self.pos[1:] + velocity[1:] * self.friction + total_acc * dt * dt

        self.prev_pos = last_pos

        # 因為我們在第1步強行移動了 pos[0]，prev_pos[0] 也需要跟上，
        # 否則計算會有誤差，但在純位置約束法中這行其實非必要，主要是為了邏輯一致性
        self.prev_pos[0] = current_head_pos

        # --- 3. 距離約束 (Constraints) ---
        # 這是波動傳遞的關鍵
        for _ in range(20):  # 增加迭代次數讓傳遞更硬實
            self.apply_constraints()

    def apply_constraints(self):
        # 這裡不需要再把 pos[0] 歸零了，因為我們在 update 裡已經設定了它的動態位置

        for i in range(self.n - 1):
            dist_vec = self.pos[i + 1] - self.pos[i]
            dist = np.linalg.norm(dist_vec)

            if dist == 0:
                continue

            correction_factor = (dist - self.segment_len) / dist
            correction_vec = dist_vec * correction_factor * 0.5

            # 邏輯判斷：
            # 第 0 點是 "無限大質量" (由我們的手控制)，所以它絕對不移動。
            # 第 i+1 點必須承受所有的修正量。
            if i == 0:
                self.pos[i + 1] -= 2 * correction_vec
            else:
                # 其他節點之間則互相拉扯 (各移動一半)
                self.pos[i] += correction_vec
                self.pos[i + 1] -= correction_vec


# --- 視覺化設定 ---

sim = ShakingRopeSimulation(n_segments=40, total_length=15.0)
fig, ax = plt.subplots(figsize=(6, 8))

ax.set_xlim(-10, 10)
ax.set_ylim(-18, 2)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Moving Anchor (Wave Propagation)")

# 畫繩子
(line,) = ax.plot([], [], "o-", lw=2, markersize=3, color="tab:orange")
# 畫出 "手" 的位置 (紅色大點)
(hand_dot,) = ax.plot([], [], "o", color="red", markersize=8)


def init():
    line.set_data([], [])
    hand_dot.set_data([], [])
    return line, hand_dot


def animate(frame):
    dt = 0.016
    current_time = frame * dt

    sim.update(dt, current_time)

    line.set_data(sim.pos[:, 0], sim.pos[:, 1])
    hand_dot.set_data([sim.pos[0, 0]], [sim.pos[0, 1]])  # 修正這裡的數據格式
    return line, hand_dot


ani = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=16, blit=True)

plt.show()
