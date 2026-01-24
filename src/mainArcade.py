import os
import threading
from Const import *
import arcade
from tracker import AsyncFaceTracker, FakeTracker
from ValueUtils import *
from mesh_renderer import GridMesh
import numpy as np
from live2d import Live2DPart,create_live2dpart,create_live2dpart_each

class Live2DEngine(arcade.Window):
    def __init__(self, tracker):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.tracker = tracker
        self.calibration = (-0.5867841357630763, -0.5041574138173885)
        arcade.set_background_color(arcade.color.GREEN)
        self.all_sprites = arcade.SpriteList()
        self.face_info = None
        self.lock = threading.Lock()
        pub.subscribe(self.notify_face_info, "FaceInfo")

        self.hair_physics = SpringDamper(stiffness=0.01, damping=0.6)
        self.hair_physics_pendulum = PendulumPhysics(
            stiffness=0.01, damping=0.6, mass=1.0, gravity_power=1.0
        )
        self.last_yaw = 0
        self.total_time = 0
        self.camera = arcade.Camera2D()
        self.camera.position = (0, 950)
        self.camera.zoom = 1.5
        self.hair_bones = None
        self.lives, self.root = create_live2dpart(self.ctx)

    def notify_face_info(self, face_info):
        with self.lock:
            self.face_info = face_info

    def __del__(self):
        self.tracker.release()

    def on_draw(self):
        self.clear()
        self.camera.use()

        for _, obj in self.lives.items():
            obj.draw()

    def update_pose(self, face_info):
        yaw, pitch, roll = filterHead(face_info["Pose"])
        face_info["Yaw"] = yaw
        face_info["Pitch"] = pitch
        face_info["Roll"] = roll

    def on_update(self, delta_time):
        with self.lock:
            face_info = self.face_info
        if face_info is None:
            return
        self.total_time += delta_time
        self.update_pose(face_info)
        self.root.update(face_info)

class TestMesh(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Level 3: Mesh Test")
        arcade.set_background_color(arcade.color.GREEN)

        # 這裡不使用 arcade.SpriteList，因為我們是自繪幾何
        # 初始化 GridMesh (建議把 front_hair 路徑換上去)
        self.hair = create_live2dpart_each(self.ctx,"FrontHairLeft", None )
        self.camera = arcade.Camera2D()
        self.camera.position = (0, 0)
        # ... (初始化代碼) ...
        self.total_time = 0.0

    def on_draw(self):
        self.clear()
        self.camera.use()
        # 直接呼叫我們寫的 draw
        self.hair.draw()

    def on_update(self, delta_time):
        self.total_time += delta_time

if __name__ == "__main__":
    # 載入設定檔
    #game = Live2DEngine(tracker=FakeTracker())
    #game = MyGame(tracker=AsyncFaceTracker())
    game = TestMesh()
    arcade.run()
