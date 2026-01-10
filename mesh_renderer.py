import arcade
import arcade.gl
import numpy as np
from array import array
import arcade
import math
from bone_system import Bone


class GridMesh:
    def __init__(
        self,
        context: arcade.context,
        texture_path: str,
        grid_size=(100, 100),
        scale=1.0,
        parent=None,
        data_key=None,
    ):
        self.ctx = context

        # 1. 載入紋理
        self.texture = self.ctx.load_texture(texture_path)
        self.width = self.texture.width * scale
        self.height = self.texture.height * scale

        # 網格密度 (Cols, Rows)
        self.cols, self.rows = grid_size

        # 2. 定義 Shader (這是最基礎的 Pass-through Shader)
        # 頂點著色器：負責計算點的位置 (之後變形就在這裡做手腳)
        vertex_source = """
        #version 330
        
        // 輸入屬性
        in vec2 in_vert;
        in vec2 in_uv;
        
        // 輸出給 Fragment Shader 的變數
        out vec2 v_uv;
        
        // 全域變數 (Uniforms)
        uniform vec2 u_pos;     // 網格整體位置
        uniform mat4 u_proj;    // 投影矩陣 (把世界座標轉為螢幕座標)

        void main() {
            v_uv = in_uv;
            // 計算最終位置: 原始頂點 + 整體位移
            vec2 final_pos = in_vert + u_pos;
            gl_Position = u_proj * vec4(final_pos, 0.0, 1.0);
        }
        """

        # 片段著色器：負責填色
        fragment_source = """
        #version 330
        
        in vec2 v_uv;
        out vec4 f_color;
        
        uniform sampler2D u_texture;

        void main() {
            // 從紋理取樣顏色
            f_color = texture(u_texture, v_uv);
        }
        """

        self.program = self.ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=fragment_source,
        )

        # 3. 生成網格資料 (Vertices & Indices)
        self.setup_mesh_data()

        self.parent = parent
        self.children = []
        if parent:
            parent.children.append(self)

        # 儲存從 JSON 讀來的原始設定 (方便 Debug)
        self.data_key = data_key

        self.base_local_x = 0
        self.base_local_y = 0

        # 這些是 "相對" 於父物件的屬性 (Local Transform)
        self.local_scale_x = 1.0
        self.local_scale_y = 1.0
        self.local_x = 0
        self.local_y = 0
        self.local_angle = 0
        self.local_scale_y = 1.0

        # 錨點 (0.0 ~ 1.0)，預設中心
        self.anchor_x_ratio = 0.5
        self.anchor_y_ratio = 0.5

        self.center_x = 0
        self.center_y = 0

    def update_transform(self):
        """核心：遞迴更新座標"""
        if self.parent:
            # 1. 取得父物件資訊
            p_x, p_y = self.parent.center_x, self.parent.center_y
            p_angle = self.parent.angle
            p_scale = self.parent.scale

            # 2. 計算旋轉 (父角度 + 本地角度)
            # 這裡的數學確保了 "跟著爸爸轉"
            rad = math.radians(p_angle)

            # 旋轉公式
            rot_x = self.local_x * math.cos(rad) - self.local_y * math.sin(rad)
            rot_y = -(self.local_x * math.sin(rad) + self.local_y * math.cos(rad))

            # 3. 更新自己的全域座標
            self.center_x = p_x + rot_x * p_scale[0]
            self.center_y = p_y + rot_y * p_scale[1]
            self.angle = p_angle + self.local_angle
            self.scale = p_scale

            # 特別處理：眨眼縮放 (Y軸)
            self.height = self.texture.height * self.scale[1] * self.local_scale_y

            # TODO: 這裡尚未實作 "自身錨點旋轉" (Self-Pivot)，
            # 目前旋轉是以 Sprite 中心為主。如果要讓頭部繞著脖子轉，
            # 需要再加一段 offset math，但 MVP 先這樣即可。
        else:
            self.center_x = self.local_x
            self.center_y = self.local_y

        # 遞迴更新孩子
        for child in self.children:
            child.update_transform()

    def setup_mesh_data(self):
        """
        生成 N x M 的網格頂點數據
        Data Layout: [x, y, u, v, x, y, u, v, ...]
        """
        vertices = []
        indices = []

        # 步驟 A: 生成頂點 (Vertices)
        # 我們把圖片中心設為 (0, 0)，方便之後旋轉
        start_x = -self.width / 2
        start_y = -self.height / 2
        step_x = self.width / self.cols
        step_y = self.height / self.rows

        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                # 計算物理座標 (x, y)
                px = start_x + (c * step_x)
                py = start_y + (r * step_y)

                # 計算紋理座標 (u, v) -> 範圍 0.0 ~ 1.0
                u = c / self.cols
                v = (
                    r / self.rows
                )  # 注意：有些 OpenGL 系統 v 是反的，若圖片倒置需改成 (1 - r/rows)

                vertices.extend([px, py, u, v])

        # 步驟 B: 生成索引 (Indices) - 定義三角形
        # 每個格子 (Quad) 切成兩個三角形
        for r in range(self.rows):
            for c in range(self.cols):
                # 計算當前格子的四個頂點索引
                # i_tl = top-left, i_br = bottom-right
                i_bl = r * (self.cols + 1) + c
                i_br = i_bl + 1
                i_tl = (r + 1) * (self.cols + 1) + c
                i_tr = i_tl + 1

                # 三角形 1: BL -> BR -> TL
                indices.extend([i_bl, i_br, i_tl])
                # 三角形 2: TL -> BR -> TR
                indices.extend([i_tl, i_br, i_tr])

        # 步驟 C: 打包進 GPU Buffer
        # '2f 2f' 代表 2個 float (x,y) + 2個 float (u,v)
        self.vbo = self.ctx.buffer(data=array("f", vertices))
        self.ibo = self.ctx.buffer(data=array("I", indices))

        self.original_vertices = np.array(vertices, dtype="f4")

        # 這是我們要即時修改並傳給 GPU 的陣列
        self.current_vertices = self.original_vertices.copy()
        # 建立 VBO 時，要標記為 dynamic (雖然 arcade 預設通常沒差，但語意上比較正確)
        self.vbo = self.ctx.buffer(data=array("f", self.current_vertices))

        # 定義 Geometry 物件
        self.geometry = self.ctx.geometry(
            [arcade.gl.BufferDescription(self.vbo, "2f 2f", ["in_vert", "in_uv"])],
            index_buffer=self.ibo,
            mode=self.ctx.TRIANGLES,
        )

    def update_buffer(self):
        """將 current_vertices 的內容寫入 GPU"""
        # 注意：這是二進位寫入，速度很快
        self.vbo.write(self.current_vertices.tobytes())

    def apply_bend(self, bend_amount, time_offset=0.0):
        """
        對網格進行彎曲變形
        bend_amount: 浮點數，負值向左彎，正值向右彎
        """
        # 1. 重置為原始形狀 (Reshape 成 N x 4，方便操作 X, Y)
        # 每一列是 [x, y, u, v]
        data = self.original_vertices.copy().reshape(-1, 4)

        # 取得所有點的 Y 座標
        ys = data[:, 1]

        # 2. 計算權重 (Weight)
        # 假設圖片中心是 (0,0)，上半部是頭髮根部，下半部是髮尾
        # 我們要把 Y 座標正規化，讓根部不動 (weight=0)，髮尾動最多 (weight=1)

        # 找出網格的頂部和底部邊界
        top_y = ys.max()
        bottom_y = ys.min()
        height = top_y - bottom_y

        if height == 0:
            return

        # 3. 計算每個點的權重與角度
        # 根部 (top_y) weight = 0 (不動)
        # 髮尾 (bottom_y) weight = 1 (轉最多)
        current_y = data[:, 1]
        normalized_y = weight = (top_y - current_y) / height
        # 使用平方曲線讓彎曲更自然 (根部比較硬，髮尾比較軟)
        weight = weight**2

        # 計算旋轉角度 (弧度制)
        # 負號是用來修正方向 (依據你的座標系可能需要調整)
        base_theta = -bend_amount * weight
        wave_theta = np.sin(normalized_y * 5.0 - time_offset * 2.0) * 0.01 * weight
        theta = base_theta + wave_theta

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 4. 執行旋轉矩陣 (Rotation Matrix)
        # 公式：
        # x' = x * cos - y * sin
        # y' = x * sin + y * cos
        # 但我們必須繞著「根部」轉，所以要先移到相對座標，轉完再移回來

        # 暫時將 Y 軸原點設為根部
        rel_x = data[:, 0]  # 假設 X 中心原本就是 0
        rel_y = data[:, 1] - top_y  # 讓根部變成 0

        # 套用矩陣
        new_x = rel_x * cos_t - rel_y * sin_t
        new_y = rel_x * sin_t + rel_y * cos_t

        # 移回世界座標
        data[:, 0] = new_x  # + self.center_x
        data[:, 1] = new_y + top_y  # + self.center_y  # 防呆
        print(data[-1])

        ## 正規化：Top = 0.0, Bottom = 1.0
        ## (這裡假設 Y 向上為正，如果你的座標系 Y 向下為正，公式要反過來)
        # normalized_y = (top_y - ys) / height

        ## 3. 應用變形公式： x_offset = bend_amount * (y_ratio ^ 2)
        ## 使用平方 (square) 會讓彎曲看起來更像拋物線，比較自然
        # offsets = bend_amount * (normalized_y**2) * 100  # *100 是放大係數，讓效果明顯點

        # 更新 X 座標
        # data[:, 0] += offsets + self.center_x
        # data[:, 1] += self.center_y

        # 4. 存回 current_vertices 並上傳 GPU
        self.current_vertices = data.flatten()
        self.update_buffer()

    def draw(self):
        # 1. 【關鍵修正】開啟混合模式
        # 這行指令告訴 GPU：計算像素時，要考慮 Alpha 通道
        self.ctx.enable(self.ctx.BLEND)

        # 2. 設定混合方程式 (通常 Arcade 預設就是這個，但為了保險起見可以明確指定)
        # source (你的圖) * alpha + destination (背景) * (1 - alpha)
        self.ctx.blend_func = self.ctx.BLEND_DEFAULT

        # 3. 綁定資源與渲染
        self.texture.use(0)
        self.program["u_pos"] = (self.center_x, self.center_y)
        self.program["u_proj"] = self.ctx.projection_matrix
        self.program["u_texture"] = 0

        self.geometry.render(self.program)


import numpy as np


class SkinnedMesh(GridMesh):
    def __init__(self, context, texture_path, bones, grid_size=(10, 10), scale=1.0):
        # 1. 先像以前一樣建立網格
        super().__init__(context, texture_path, grid_size, scale)

        self.bones = bones  # 骨骼列表 [Bone, Bone, ...]

        # 2. 儲存綁定資訊
        # vertex_indices: 紀錄第 i 個頂點屬於第幾號骨頭
        # local_positions: 紀錄第 i 個頂點在該骨頭空間的相對座標 (Bone Space)
        self.vertex_bone_indices = []
        self.vertex_local_positions = []

        # 3. 執行綁定 (Bind Pose Calculation)
        self.bind_mesh_to_bones()

    def bind_mesh_to_bones(self):
        # 準備數據
        vertices = self.original_vertices.copy().reshape(-1, 4)

        # 1. 找出網格的垂直範圍 (Y-Range)
        # 假設 Y 軸向上為正 (Arcade 預設)，Top Y 是最大值
        all_ys = vertices[:, 1]
        max_y = np.max(all_ys)  # 髮根位置
        min_y = np.min(all_ys)  # 髮尾位置
        height = max_y - min_y

        # 確保骨頭是最新的
        for b in self.bones:
            b.update()

        self.vertex_weights = []
        self.vertex_local_positions = []

        print("--- 使用垂直線性權重 (Linear Gradient) ---")

        for i, _ in enumerate(vertices):
            vx, vy = vertices[i][0] + self.center_x, vertices[i][1] + self.center_y

            # --- 核心修改開始 ---

            # 2. 計算該頂點在垂直高度上的比例 (0.0 ~ 1.0)
            # 1.0 = 最頂端 (Root), 0.0 = 最底端 (Tip)
            if height > 0:
                ratio = (vy - min_y) / height
            else:
                ratio = 0.5  # 防呆

            # 3. 強制分配權重
            # 假設 bones[0] 是 Root, bones[1] 是 Tip
            # 我們可以加一點 "Power" 讓過渡更硬一點，避免中間太軟
            # ratio = ratio ** 0.5 (如果想要髮根影響範圍更大)

            w_root = ratio

            # 儲存權重 (Root index=0, Tip index=1)
            final_weights = [(0, w_root)]

            # --- 核心修改結束 ---

            self.vertex_weights.append(final_weights)

            # 4. 計算相對座標 (這部分跟之前一樣)
            local_pos_map = {}
            v_vec = np.array([vx, vy, 1.0])

            for b_idx, w in final_weights:
                target_bone = self.bones[b_idx]
                inv_matrix = np.linalg.inv(target_bone.world_matrix)
                v_local = inv_matrix @ v_vec
                local_pos_map[b_idx] = (v_local[0], v_local[1])

            self.vertex_local_positions.append(local_pos_map)

    def update_skinning(self):
        """
        每一幀呼叫：根據骨頭的新位置，更新網格頂點
        """
        # 準備寫入 GPU 的數據
        new_data = self.original_vertices.copy().reshape(-1, 4)

        # 遍歷所有頂點 (這是 CPU Skinning，頂點多會慢，Level 5 會移到 Shader)
        for i, I in enumerate(new_data):
            weights_info = self.vertex_weights[i]  # [(idx1, w1), (idx2, w2)]
            local_pos_map = self.vertex_local_positions[i]  # {idx1: pos1, idx2: pos2}

            final_x = 0.0
            final_y = 0.0

            # 混合所有骨頭的影響
            for b_idx, weight in weights_info:
                bone = self.bones[b_idx]
                m = bone.world_matrix
                lx, ly = local_pos_map[b_idx]

                # 算出如果只跟隨這根骨頭，頂點會在哪
                # V_world = M * V_local
                wx = m[0, 0] * lx + m[0, 1] * ly + m[0, 2]
                wy = m[1, 0] * lx + m[1, 1] * ly + m[1, 2]

                # 累加 (Weighted Sum)
                final_x += wx * weight
                final_y += wy * weight

            # 寫回 Buffer (轉回相對座標)
            new_data[i][0] = final_x  # + self.center_x
            new_data[i][1] = final_y  # + self.center_y

        # 上傳 GPU
        self.current_vertices = new_data.flatten()
        self.update_buffer()

    def recalculate_local_positions(self):
        """
        Recalculates the local position of vertices relative to their bound bones.
        Must be called after loading weights or modifying weights in the editor.
        Assuming the mesh and bones are currently in 'Bind Pose' (T-pose).
        """
        # 1. Reset the storage
        self.vertex_local_positions = []

        # Get raw vertex data (x, y, u, v)
        vertices = self.original_vertices.copy().reshape(-1, 4)

        # Ensure bone matrices are up-to-date
        for b in self.bones:
            b.update()

        # 2. Iterate through all vertices
        for i, _ in enumerate(vertices):
            # Calculate Vertex World Position
            # We must add self.x/y because original_vertices are local to the mesh center
            vx = vertices[i][0] + self.center_x
            vy = vertices[i][1] + self.center_y

            # Prepare vector for matrix multiplication [x, y, 1]
            v_vec = np.array([vx, vy, 1.0])

            # Get the weights for this vertex: [(bone_idx, weight), ...]
            weights_info = self.vertex_weights[i]

            # Dictionary to store relative pos for each influencing bone
            # Format: {bone_idx: (local_x, local_y)}
            local_pos_map = {}

            for b_idx, weight in weights_info:
                target_bone = self.bones[b_idx]

                # Formula: V_local = Inverse(Bone_World_Matrix) * V_world
                # Calculate inverse matrix of the bone
                try:
                    inv_matrix = np.linalg.inv(target_bone.world_matrix)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrix (rare in 2D transform)
                    inv_matrix = np.identity(3)

                # Transform world coordinate to bone local coordinate
                v_local = inv_matrix @ v_vec

                # Store it
                local_pos_map[b_idx] = (v_local[0], v_local[1])

            self.vertex_local_positions.append(local_pos_map)

        print(f"Recalculated local positions for {len(vertices)} vertices.")


test_path = "assets/sample_model/processed/HairFront.png"


class MyWindow(arcade.Window):
    def __init__(self):
        super().__init__(800, 600, "Level 3: Mesh Test")
        arcade.set_background_color(arcade.color.GREEN)

        self.root = Bone("Root", y=0)
        self.tip = Bone("Tip", y=-100, parent=self.root)

        # 這裡不使用 arcade.SpriteList，因為我們是自繪幾何
        # 初始化 GridMesh (建議把 front_hair 路徑換上去)
        self.hair_mesh = SkinnedMesh(
            self.ctx,
            texture_path=test_path,
            grid_size=(10, 10),
            scale=1.0,
            bones=[self.root, self.tip],
        )
        SCREEN_WIDTH = 800
        SCREEN_HEIGHT = 600

        self.x = self.hair_mesh.center_x = SCREEN_WIDTH / 2
        self.y = self.hair_mesh.center_y = SCREEN_HEIGHT / 2
        self.hair_mesh.x = self.hair_mesh.center_x
        self.hair_mesh.y = self.hair_mesh.center_y
        # ... (初始化代碼) ...
        self.total_time = 0.0

    def on_draw(self):
        self.clear()
        # 直接呼叫我們寫的 draw
        self.hair_mesh.draw()
        # Debug: 畫出骨頭節點
        for bone in self.root, self.tip:
            wx, wy = bone.get_world_position()
            wx += self.x
            wy += self.y
            arcade.draw_circle_filled(wx, wy, 5, arcade.color.RED)

            if bone.parent:
                px, py = bone.parent.get_world_position()
                px += self.x
                py += self.y
                arcade.draw_line(wx, wy, px, py, arcade.color.YELLOW, 2)
            else:
                px = self.x
                py = self.y
                arcade.draw_line(wx, wy, px, py, arcade.color.YELLOW, 2)

    def on_update(self, delta_time):
        self.total_time += delta_time

        # 產生一個來回擺動的數值 (-1.0 ~ 1.0)
        bend_value = np.sin(self.total_time * 3.0)
        self.root.angle = bend_value * 10.0
        self.tip.angle = 5 * bend_value * 10.0
        self.root.update()
        self.tip.update()
        # 呼叫我們剛寫的彎曲函式
        # self.hair_mesh.apply_bend(bend_value)
        self.hair_mesh.update_skinning()


if __name__ == "__main__":
    window = MyWindow()
    arcade.run()
