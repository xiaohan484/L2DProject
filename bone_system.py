import numpy as np
import json


def load_bones(filepath):
    """從 JSON 讀取骨骼並重建父子關係"""
    with open(filepath, "r") as f:
        data = json.load(f)

    bones = []
    bone_map = {}  # name -> Bone Object

    # 第一輪：建立物件
    parent_x = 0
    parent_y = 0
    for b_data in data:
        bone = Bone(b_data["name"], b_data["x"], b_data["y"])
        bones.append(bone)
        bone_map[bone.name] = bone

    # 第二輪：建立連結
    for b_data in data:
        if b_data["parent"]:
            child = bone_map[b_data["name"]]
            parent = bone_map[b_data["parent"]]
            child.parent = parent
            parent.children.append(child)
    for b in bones:
        b.update()
    return bones


class Bone:
    def __init__(self, name, x=0, y=0, angle=0, scale_x=1, scale_y=1, parent=None):
        self.name = name
        self.parent = parent  # 父骨骼引用
        self.children = []  # 子骨骼列表

        if parent:
            parent.children.append(self)

        # === Local Transform (本地屬性) ===
        # 這些是你每一幀會去修改的數值 (例如透過物理或 PnP)
        self.x = x
        self.y = y
        self.angle = angle  # 單位：度 (Degree)
        self.scale_x = scale_x
        self.scale_y = scale_y

        # === Matrices (快取矩陣) ===
        # Local Matrix: 描述我相對於爸爸的變換
        self.local_matrix = np.identity(3)
        # World Matrix: 描述我相對於世界原點 (0,0) 的變換
        self.world_matrix = np.identity(3)
        self.update()

    def update(self):
        """
        核心函式：計算矩陣並遞迴更新子骨骼
        """
        # 1. 計算 Local Matrix (SRT 順序)
        # 先轉弧度
        rad = np.radians(self.angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        # 為了效能，我們直接手寫矩陣乘法結果 (比用 np.dot 三次快)
        # M = T * R * S
        self.local_matrix = np.array(
            [
                [self.scale_x * cos_a, -self.scale_y * sin_a, self.x],
                [self.scale_x * sin_a, self.scale_y * cos_a, self.y],
                [0, 0, 1],
            ]
        )
        # 2. 計算 World Matrix (Forward Kinematics)
        if self.parent:
            # 我的世界座標 = 爸爸的世界矩陣 x 我的本地矩陣
            # 注意：在 NumPy 裡，如果向量是行向量(Column Vector)，通常寫作 M1 @ M2
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            # 我是根節點，我的世界就是本地
            self.world_matrix = self.local_matrix
        # 3. 遞迴更新所有小孩 (Propagation)
        for child in self.children:
            child.update()

    def get_world_position(self):
        """
        輔助函式：從矩陣中萃取世界座標 (x, y)
        矩陣的第三行 (Column 2) 就是位移向量
        """
        return self.world_matrix[0, 2], self.world_matrix[1, 2]


def test1():
    # === 測試場景 ===
    # 建立一個手臂結構：肩膀 (Root) -> 手肘 (Child) -> 手掌 (Grandchild)
    print("--- 初始化骨骼 ---")
    # 肩膀在世界座標 (100, 100)
    shoulder = Bone("Shoulder", x=100, y=100)
    # 手肘在肩膀的右邊 50 單位 (相對座標 50, 0)
    elbow = Bone("Elbow", x=50, y=0, parent=shoulder)
    # 手掌在手肘的右邊 30 單位 (相對座標 30, 0)
    hand = Bone("Hand", x=30, y=0, parent=elbow)
    # 強制更新一次矩陣
    shoulder.update()
    print(f"肩膀世界座標: {shoulder.get_world_position()} (預期: 100, 100)")
    print(f"手肘世界座標: {elbow.get_world_position()}    (預期: 150, 100)")
    print(f"手掌世界座標: {hand.get_world_position()}     (預期: 180, 100)")
    print("\n--- 測試旋轉 (FK) ---")
    # 旋轉肩膀 90 度 (整隻手臂應該垂直向下指，因為螢幕座標 Y 向下通常是正的，或向上視你的系通而定)
    # 假設 Y 向下為正 (Arcade 預設 Y 向上，這裡我們先看數值)
    shoulder.angle = 90
    shoulder.update()
    # 數學預測：
    # 肩膀還是在 (100, 100)
    # 手肘應該轉到肩膀「下面」 (100, 150) (如果是 Y 軸向下增加) 或是 (100, 50) (如果 Y 向上增加)
    # 手掌應該在手肘「下面」
    wx, wy = hand.get_world_position()
    print(f"肩膀旋轉 90 度後，手掌的世界座標: ({wx:.2f}, {wy:.2f})")
    if abs(wx - 100) < 0.1:
        print(">> PASS: 手掌正確跟隨肩膀旋轉！")
    else:
        print(">> FAIL: 骨骼連動失效，請檢查矩陣乘法。")
    return


if __name__ == "__main__":
    test1()
