import json
import math
import os
from Const import *
from VTSprite import *
import arcade
from tracker import AsyncFaceTracker
from ValueUtils import *

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.tracker = AsyncFaceTracker()
        self.calibration = (-0.28554,0.1093026)

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
        filepath = os.path.join("assets/sample_model/processed", data['filename'])
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
            sprite.initial_global_x = raw_global_x
            sprite.initial_global_y = raw_global_yal_y = raw_global_y
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
            sprite.center_y = SCREEN_HEIGHT / 2 - 950 * GLOBAL_SCALE # 試著改成 +50 或 -50 看看效果
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
        self.body = self.create_vt_sprite("Body")

        # --- 2. 建立五官 (綁定在 Face 上) ---
        # 程式會自動算出它們相對於臉的 local_x/y
        self.face = self.create_vt_sprite("Face",parent=self.body) 
        #self.all_sprites.append(self.face)
        
        # 眼白 (底)
        self.eye_white_L = self.create_vt_sprite("EyeWhiteL", parent=self.face)
        self.eye_white_R = self.create_vt_sprite("EyeWhiteR", parent=self.face)
        
        # 眼珠 (中)
        self.eye_pupil_L = self.create_vt_sprite("EyePupilL", parent=self.face)
        self.eye_pupil_R = self.create_vt_sprite("EyePupilR", parent=self.face)

        self.eye_lid_L = self.create_vt_sprite("EyeLidL", parent=self.face)
        self.eye_lid_R = self.create_vt_sprite("EyeLidR", parent=self.face)
        
        # 睫毛 (上)
        self.eye_lash_L = self.create_vt_sprite("EyeLashL", parent=self.face)
        self.eye_lash_R = self.create_vt_sprite("EyeLashR", parent=self.face)

        #
        self.eye_brow_L = self.create_vt_sprite("EyeBrowL", parent=self.face)
        self.eye_brow_R = self.create_vt_sprite("EyeBrowR", parent=self.face)
        
        # 前髮 (最上層)
        self.hair_front = self.create_vt_sprite("HairFront", parent=self.face)
    def on_draw(self):
        self.clear()
        self.all_sprites.draw()
    def on_key_release(self, key,modifiers):
        if key == arcade.key.C:
            self.calibration = self.tracker.get_iris_pos()
            print(self.calibration)
    def update_body(self):
        # 呼吸頻率 (速度)
        BREATH_SPEED = 1.5
        # 呼吸幅度 (縮放比例，不用太大，0.01 代表 1%)
        BREATH_AMOUNT = 0.0025
        import time
        # 利用時間算出一個 -1 ~ 1 的波形
        breath_wave = math.sin(time.time() * BREATH_SPEED)
        # 應用在身體 (Root) 的 Y 軸縮放
        # 1.0 是原始大小，加上波形變化
        self.body.scale_y = GLOBAL_SCALE + breath_wave * BREATH_AMOUNT
        self.body.scale_x = GLOBAL_SCALE + breath_wave * (BREATH_AMOUNT * 0.1) # X軸稍微跟著動一點點會更自然
        return
    def update_pose(self):
        yaw,pitch,roll = filterHead(self.tracker.get_head_pose())
        # 2. 定義強度 (這就是你要一直調的參數)
        # 代表轉 1 度，像素要移動多少 px
        PARALLAX_X_STRENGTH = 0.25
        PARALLAX_Y_STRENGTH = 0.5

        # 3. 計算各層的位移量 (Offset)
        # Layer Depth Multipliers (深度乘數)
        # 前面動得快，後面動得慢
        OFFSET_FRONT = 1.0001  # 前髮、五官
        OFFSET_MID   = 1.0     # 臉型
        OFFSET_BACK  = 0.9999  # 後髮

        # 計算 X 軸位移
        move_x_front = yaw * PARALLAX_X_STRENGTH * OFFSET_FRONT
        move_x_mid   = yaw * PARALLAX_X_STRENGTH * OFFSET_MID
        move_x_back  = yaw * PARALLAX_X_STRENGTH * OFFSET_BACK
        # 計算 Y 軸位移
        move_y_front = pitch * PARALLAX_Y_STRENGTH * OFFSET_FRONT
        move_y_mid   = pitch * PARALLAX_Y_STRENGTH * OFFSET_MID
        move_y_back  = pitch * PARALLAX_Y_STRENGTH * OFFSET_BACK

        # 4. 應用到 Sprites (記得加上原本的基礎位置)
        # 假設 base_x 是螢幕中心
        self.face_features_center_x = move_x_front
        self.face_base_center_x     = move_x_mid
        self.back_hair_center_x     = move_x_back

        self.face_features_center_y = move_y_front
        self.face_base_center_y     = move_y_mid
        self.back_hair_center_y     = move_y_back

        self.body.local_x = self.body.base_local_x + move_x_back
        self.face.local_x = self.face.base_local_x + move_x_mid
        self.eye_white_L.local_x = self.eye_white_L.base_local_x + move_x_front
        self.eye_white_R.local_x = self.eye_white_R.base_local_x + move_x_front
        self.eye_lid_L.local_x   =   self.eye_lid_L.base_local_x + move_x_front  
        self.eye_lid_R.local_x   =   self.eye_lid_R.base_local_x + move_x_front  
        self.eye_lash_L.local_x  =  self.eye_lash_L.base_local_x + move_x_front 
        self.eye_lash_R.local_x  =  self.eye_lash_R.base_local_x + move_x_front 
        self.eye_brow_L.local_x  =  self.eye_brow_L.base_local_x + move_x_front 
        self.eye_brow_R.local_x  =  self.eye_brow_R.base_local_x + move_x_front 
        self.hair_front.local_x  =  self.hair_front.base_local_x + move_x_front 

        self.body.local_y = self.body.base_local_y + move_y_back
        self.face.local_y = self.face.base_local_y + move_y_mid
        self.eye_white_L.local_y = self.eye_white_L.base_local_y + move_y_front
        self.eye_white_R.local_y = self.eye_white_R.base_local_y + move_y_front
        self.eye_lid_L.local_y   =   self.eye_lid_L.base_local_y + move_y_front  
        self.eye_lid_R.local_y   =   self.eye_lid_R.base_local_y + move_y_front  
        self.eye_lash_L.local_y  =  self.eye_lash_L.base_local_y + move_y_front 
        self.eye_lash_R.local_y  =  self.eye_lash_R.base_local_y + move_y_front 
        self.eye_brow_L.local_y  =  self.eye_brow_L.base_local_y + move_y_front 
        self.eye_brow_R.local_y  =  self.eye_brow_R.base_local_y + move_y_front 
        self.hair_front.local_y  =  self.hair_front.base_local_y + move_y_front 

    def on_update(self, delta_time):
        self.update_body()
        self.update_pose()


        blinkL, blinkR = filterBlink(self.tracker.get_eye_blink_ratio())
        is_blinking = (blinkL < 0.25) or (blinkR < 0.25)
        final_x, final_y = convertPupils(self.tracker.get_iris_pos(), self.calibration, is_blinking)
        final_x += self.face_features_center_x
        final_y += self.face_features_center_y
        # pupils
        self.eye_pupil_L.local_x = self.eye_pupil_L.base_local_x + final_x
        self.eye_pupil_R.local_x = self.eye_pupil_R.base_local_x + final_x
        self.eye_pupil_L.local_y = self.eye_pupil_L.base_local_y + final_y
        self.eye_pupil_R.local_y = self.eye_pupil_R.base_local_y + final_y

        #EyeLid
        target_y_L = map_range(blinkL, EAR_MIN, EAR_MAX, EYE_CLOSED_Y, EYE_OPEN_Y)
        target_y_R = map_range(blinkR, EAR_MIN, EAR_MAX, EYE_CLOSED_Y, EYE_OPEN_Y)
        target_y_L += self.face_features_center_y
        target_y_R += self.face_features_center_y
        self.eye_lid_L.local_y = self.eye_lid_L.base_local_y + target_y_L
        self.eye_lid_R.local_y = self.eye_lid_R.base_local_y + target_y_R
        self.eye_lash_L.local_y = self.eye_lash_L.base_local_y + target_y_L
        self.eye_lash_R.local_y = self.eye_lash_R.base_local_y + target_y_R

        close_progressL = map_range(blinkL, EAR_MIN, EAR_MAX, 1.0, 0.0)
        close_progressR = map_range(blinkR, EAR_MIN, EAR_MAX, 1.0, 0.0)
        target_scale_y_L = 1.0 - (close_progressL *0.4) 
        target_scale_y_R = 1.0 - (close_progressR *0.4) 
        #print(target_scale_y_L,target_scale_y_R)

        if blinkL < EAR_MIN:
            self.eye_lash_L.local_scale_y = -1 * target_scale_y_L
            self.eye_white_L.local_scale_y = 0
            self.eye_pupil_L.local_scale_y = 0
        else:
            self.eye_lash_L.local_scale_y = 1 * target_scale_y_L
            self.eye_white_L.local_scale_y = 1
            self.eye_pupil_L.local_scale_y = 1

        if blinkR < EAR_MIN:
            self.eye_lash_R.local_scale_y = -1 * target_scale_y_R
            self.eye_white_R.local_scale_y = 0
            self.eye_pupil_R.local_scale_y = 0
        else:
            self.eye_lash_R.local_scale_y = 1 *target_scale_y_R
            self.eye_white_R.local_scale_y = 1
            self.eye_pupil_R.local_scale_y = 1
        # 記得觸發更新
        self.body.update_transform()
        

if __name__ == "__main__":
    # 載入設定檔
    game = MyGame()
    arcade.run()