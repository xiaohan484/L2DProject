def map_range(value, in_min, in_max, out_min, out_max):
    """
    將數值從一個範圍映射到另一個範圍。
    例如：將 blink_ratio (0.2~0.5) 映射到 local_y (0~25)
    """
    # 先算出正規化比例 (0.0 ~ 1.0)
    norm = (value - in_min) / (in_max - in_min)
    # 限制在 0.0 ~ 1.0 之間 (Clamping)
    norm = max(0.0, min(1.0, norm))
    # 映射到輸出範圍
    return out_min + norm * (out_max - out_min)


class SpringDamper:
    def __init__(self, stiffness=0.1, damping=0.5):
        """
        stiffness (k): 剛性。越大越硬，回彈越快。 (建議 0.05 ~ 0.2)
        damping (d): 阻尼。越大越黏，越不容易晃個不停。 (建議 0.3 ~ 0.8)
        """
        self.stiffness = stiffness
        self.damping = damping

        self.position = 0.0  # 當前的彎曲值 (bend_amount)
        self.velocity = 0.0  # 當前的變形速度
        self.target = 0.0  # 目標值 (通常是 0，代表靜止回正)

    def update(self, perturbation):
        """
        perturbation: 外部施加的擾動 (例如頭部的轉動速度)
        """
        # 1. 施加外力 (慣性原理：頭往左轉，頭髮受力往右，所以是減號或反向)
        self.velocity += perturbation

        # 2. 彈簧物理 (Hooke's Law + Damping)
        # Force = -k * pos - d * vel
        force = -self.stiffness * self.position - self.damping * self.velocity

        # 3. 更新狀態
        self.velocity += force
        self.position += self.velocity

        return self.position
