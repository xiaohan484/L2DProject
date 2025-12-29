import json
import math
import os
from Const import *
from VTSprite import *
import arcade
from tracker import AsyncFaceTracker

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.tracker = AsyncFaceTracker()
        self.calibration = (0,0)

        arcade.set_background_color(arcade.color.GREEN)
        self.all_sprites = arcade.SpriteList()
        self.setup_scene()
    def __del__(self):
        self.tracker.release()

    def create_vt_sprite(self, name, parent=None, append=True) -> VTSprite:
        """
        【工廠函數】讀取 JSON 並自動組裝 Sprite
        name: 對應 JSON 裡的 key (例如 'face', 'eye_lash_L')
        parent: 父物件 Sprite
        """
        if name not in MODEL_DATA:
            print(f"⚠️ 警告: JSON 裡找不到 '{name}'")
            return arcade.Sprite() # 回傳空物件避免當機

        data = MODEL_DATA[name]
        filepath = os.path.join("assets/processed", data['filename'])
        # 1. 建立 Sprite
        sprite = VTSprite(filepath, scale=GLOBAL_SCALE, parent=parent, data_key=name) # scale 可全域調整
        
        # 2. 讀取錨點設定
        sprite.anchor_x_ratio = data.get('anchor_x', 0.5)
        sprite.anchor_y_ratio = data.get('anchor_y', 0.5)

        # 3. 處理座標系轉換 (最重要的一步！)
        # Krita (Y向下) -> Arcade (Y向上)
        raw_global_x = data['global_center_x']
        # 注意：這裡要把 Y 軸翻轉過來
        raw_global_y = data['global_center_y']

        # 4. 計算 Local Position (相對位置)
        if parent:
            # 如果有爸爸，我的位置 = 我的絕對位置 - 爸爸的絕對位置
            # 這樣不管爸爸之後跑到哪，我們的相對距離都保持不變
            #sprite.local_x = raw_global_x - parent.initial_global_x
            #sprite.local_y = raw_global_y - parent.initial_global_y
            # 計算出初始的相對位置
            base_x = raw_global_x - parent.initial_global_x
            base_y = raw_global_y - parent.initial_global_y
            # [修改] 設定 Base 和 Initial Local
            sprite.base_local_x = base_x
            sprite.base_local_y = base_y
            sprite.local_x = base_x
            sprite.local_y = base_y
        else:
            ## 如果是根節點 (Root)，直接放在畫面中間 (或你想要的位置)
            ## 這裡我們稍微 hack 一下，把 JSON 裡的座標 mapping 到螢幕中心
            #sprite.center_x = SCREEN_WIDTH / 2
            #sprite.center_y = SCREEN_HEIGHT / 2 - 200 # 往下移一點才看得到頭
            ## 記錄初始絕對座標，給孩子們參考用
            #sprite.initial_global_x = raw_global_x
            #sprite.initial_glob

            # 【修改點 2 & 3】重新定位根節點 (Root)
            # 如果是根節點 (臉)，把它放在視窗的正中心
            sprite.center_x = SCREEN_WIDTH / 2
            # Y 軸通常需要微調。
            # 如果設為 SCREEN_HEIGHT / 2，臉的正中心會在畫面正中心。
            # 如果想讓下巴多露出一點，可以減去一個數字 (例如 -50)
            # 如果想讓頭頂多露出一點，可以加上一個數字 (例如 +50)
            sprite.center_y = SCREEN_HEIGHT / 2  # 試著改成 +50 或 -50 看看效果
            # 記錄初始絕對座標 (這兩行不變)
            sprite.initial_global_x = raw_global_x
            sprite.initial_global_y = raw_global_yal_y = raw_global_y
        if append:
            self.all_sprites.append(sprite)

        return sprite

    def setup_scene(self):
        # --- 1. 建立根節點 (Face) ---
        # 假設你的 JSON 裡臉部叫做 "face" (如果不一樣請修改 key)
        # body
        self.face = self.create_vt_sprite("Face",append=False) 
        self.body = self.create_vt_sprite("Body",parent=self.face)
        self.all_sprites.append(self.face)

        # --- 2. 建立五官 (綁定在 Face 上) ---
        # 程式會自動算出它們相對於臉的 local_x/y
        
        # 眼白 (底)
        self.eye_white_L = self.create_vt_sprite("EyeWhiteL", parent=self.face)
        self.eye_white_R = self.create_vt_sprite("EyeWhiteR", parent=self.face)
        
        # 眼珠 (中)
        self.eye_pupil_L = self.create_vt_sprite("EyePupilL", parent=self.face)
        self.eye_pupil_R = self.create_vt_sprite("EyePupilR", parent=self.face)
        
        # 睫毛 (上)
        self.eye_lash_L = self.create_vt_sprite("EyeLashL", parent=self.face)
        self.eye_lash_R = self.create_vt_sprite("EyeLashR", parent=self.face)

        #
        self.eye_lash_L = self.create_vt_sprite("EyeBrowL", parent=self.face)
        self.eye_lash_R = self.create_vt_sprite("EyeBrowR", parent=self.face)
        
        # 前髮 (最上層)
        self.hair_front = self.create_vt_sprite("HairFront", parent=self.face)
    def on_draw(self):
        self.clear()
        self.all_sprites.draw()
    def on_key_release(self, key,modifiers):
        if key == arcade.key.C:
            self.calibration = self.tracker.get_iris_pos()
            print(self.calibration)
    def on_update(self, delta_time):
        blinkL, blinkR = self.tracker.get_eye_blink_ratio()
        is_blinking = (blinkL < 0.33) or (blinkR < 0.33)
        final_x, final_y = convertPupils(self.tracker.get_iris_pos(), self.calibration, is_blinking)
        self.eye_pupil_L.local_x = self.eye_pupil_L.base_local_x + final_x
        self.eye_pupil_R.local_x = self.eye_pupil_R.base_local_x + final_x
        self.eye_pupil_L.local_y = self.eye_pupil_L.base_local_y + final_y
        self.eye_pupil_R.local_y = self.eye_pupil_R.base_local_y + final_y
        self.face.update_transform()
        

if __name__ == "__main__":
    # 載入設定檔
    game = MyGame()
    arcade.run()