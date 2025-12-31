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