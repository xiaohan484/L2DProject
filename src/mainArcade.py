import os
import threading
from Const import *
import arcade
from tracker import AsyncFaceTracker, FakeTracker

# from ValueUtils import *
from create_mesh import create_image_mesh, PBDCloth2D
from mesh_renderer import GridMesh, CustomMesh
import numpy as np
from live2d import Live2DPart, create_live2dpart, create_live2dpart_each
from pubsub import pub
from filters import OneEuroFilter
import time

head_yaw_filter = OneEuroFilter(min_cutoff=10, beta=0.1)
head_pitch_filter = OneEuroFilter(min_cutoff=10, beta=0.1)
head_roll_filter = OneEuroFilter(min_cutoff=10, beta=0.1)


def filterHead(head_pose):
    current_time = time.time()
    yaw, pitch, roll = head_pose
    yaw = head_yaw_filter(current_time, yaw)
    pitch = head_pitch_filter(current_time, pitch)
    roll = head_roll_filter(current_time, roll)
    return yaw, pitch, roll


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
        face_info["Roll"] = 0

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

        name = "FrontHairLeft"
        # name = "FrontHairMiddle"
        # name = "BackHair"
        data = MODEL_DATA[name]
        # Assuming filename list has at least one
        filename = data["filename"][0]
        # Construct full path
        image_path = os.path.join(MODEL_PATH, filename)

        print(f"Generating mesh for {image_path}...")
        # 1. Generate Mesh using create_mesh.py logic
        # pts: (N, 2) vertices
        # tris: (M, 3) indices
        self.pts, self.tris = create_image_mesh(image_path, debug=False)

        # 2. Create CustomMesh for rendering
        self.mesh_view = CustomMesh(
            self.ctx, image_path, self.pts, self.tris, scale=GLOBAL_SCALE
        )

        # --- Initialize Physics Simulation ---
        # Extract the initial vertex positions from CustomMesh (already transformed to render coords)
        # self.mesh_view.original_vertices is flat [x, y, u, v, ...]
        render_verts = self.mesh_view.original_vertices.reshape(-1, 4)[:, :2]
        # Copy to avoid modifying original immediately
        sim_pts = render_verts.copy()

        self.sim = PBDCloth2D(sim_pts, self.tris, stiffness=0.3)
        # Adjust gravity for arcade space if needed (Y is Up, PBD gravity is (0, -9.8), matches)
        # -------------------------------------

        # 3. Wrap in Live2DPart for positioning
        # Note: MODEL_DATA coords might need adjustment depending on how create_mesh centers things
        # But Live2DPart logic should handle global positioning if we respect the anchor.

        # We need to calculate where to place it.
        # global_center_x/y in MODEL_DATA is the center of the crop in the original canvas.
        x, y = data["global_center_x"], -data["global_center_y"]

        self.hair = Live2DPart(
            name=name,
            parent=None,
            x=x,
            y=y,
            # Arcade Y is up? or we flipped it?
            # In live2d.py, it uses y=-pos[1] if parent exists.
            # If no parent, create_live2dpart_each uses x=0, y=0 and lets view handle it?
            # actually create_live2dpart_each sets x=0, y=0.
            # But the view is GridMesh centered at 0,0 local.
            # GridMesh doesn't have an offset unless we set it.
            # Let's try to match create_live2dpart_each's behavior but with explicit position?
            # Actually create_live2dpart_each sets init x=0, y=0 for root parts.
            # But here we want to place it at the specific screen location.
            # let's try 0,0 first and rely on camera.
            # Better: use the global center from data.
            z=0,
            scale_x=GLOBAL_SCALE,
            scale_y=GLOBAL_SCALE,
            view=self.mesh_view,
        )

        # Manually set position if Live2DPart(x=0,y=0) puts it at origin
        # We want it to be at global_center
        # self.hair.x = data["global_center_x"]
        # self.hair.y = -data["global_center_y"] # Try negating Y

        # But let's look at `create_live2dpart_each`:
        # For root (parent=None): x=0, y=0.
        # But the GridMesh is built with vertices relative to center.
        # Wait, GridMesh doesn't know where the "world center" is.
        # Ah, Live2DPart update() sets world_matrix.
        # If x=0, y=0, it draws at 0,0.
        # We probably want to draw it at the screen center or adjust camera.

        self.camera = arcade.Camera2D()
        self.camera.position = (x, y)  # Center camera at origin

        self.total_time = 0.0

    def on_draw(self):
        self.clear()
        self.camera.use()

        # Draw the textured mesh
        self.hair.draw()

        # Draw Wireframe (Debug)
        self.draw_wireframe()

    def draw_wireframe(self):
        # 1. Get Transform Matrix
        mat = self.hair.world_matrix  # 3x3

        # 2. Prepare vertices in Local Space (matches CustomMesh logic)
        width = self.mesh_view.texture.width
        height = self.mesh_view.texture.height
        cx = width / 2
        cy = height / 2

        # Copy pts to avoid modifying original
        local_pts = self.pts.copy()

        # Transform to CustomMesh local config:
        # px = x - center_x
        # py = (height - y) - center_y
        local_pts[:, 0] = local_pts[:, 0] - cx
        local_pts[:, 1] = (height - local_pts[:, 1]) - cy

        # 3. Apply World Transform
        # p_world = M @ p_local
        # Need to append 1 for homogeneous coords
        ones = np.ones((len(local_pts), 1))
        local_pts_h = np.hstack([local_pts, ones])

        # matrix multiplication: (3x3) @ (3xN) -> (3xN)
        # Transpose local_pts_h to (3, N)
        world_pts = mat @ local_pts_h.T
        world_pts = world_pts.T  # back to (N, 3)

        # 4. Draw Lines
        # Arcade's draw_line is slow for many lines. Use draw_lines with point list.
        # Construct line list: pairs of points

        line_points = []
        for t in self.tris:
            # Triangle p1-p2-p3
            p1 = world_pts[t[0], :2]
            p2 = world_pts[t[1], :2]
            p3 = world_pts[t[2], :2]

            # Edge 1
            line_points.extend([p1, p2])
            # Edge 2
            line_points.extend([p2, p3])
            # Edge 3
            line_points.extend([p3, p1])

        # arcade.draw_lines(line_points, arcade.color.RED, 1)

    def on_update(self, delta_time):
        self.total_time += delta_time

        # 1. Update Physics
        # The simulation is based on the "render" coordinates (centered at 0,0 locally)
        # Apply some movement to fixed points if needed (e.g. head movement simulation)
        # For now, just stepping the physics

        # Optional: Interact with mouse or movement?
        # Move the fixed points based on a fake head movement
        fixed_mask = self.sim.inv_mass == 0
        # self.sim.pos[fixed_mask, 0] += np.sin(self.total_time * 5) * 0.5

        self.sim.step()

        # 2. Update Mesh View
        # sim.pos matches the coordinate space of CustomMesh vertices [x, y]
        # We need to update the VBO

        # Get current vertex array (flat)
        # Reshape to modifying view (N, 4)
        # Note: We must not modify self.mesh_view.original_vertices directly if we want to reset
        # But here we want to update the 'current' vertices which are sent to GPU

        # Using slice assignment to update x,y only, keeping u,v
        # self.mesh_view.current_vertices is a flat float32 array

        # Reshape for easier access (this creates a view, modification reflects in original array?)
        # Numpy reshape usually returns a view, but let's be safe.
        # Actually current_vertices is 1D.

        data = self.mesh_view.current_vertices.reshape(-1, 4)
        data[:, :2] = self.sim.pos

        # Update GPU buffer
        self.mesh_view.update_buffer()

        # Sync the 'pts' used for wireframe drawing (optional, for debug view)
        # Note: 'pts' in __init__ was raw image coords.
        # But 'draw_wireframe' expects 'pts' to be in some form.
        # actually draw_wireframe does the transform: local_pts[:, 0] - cx...
        # So if we want wireframe to match sim, we should update self.pts?
        # But self.pts is (N,2) image coords. Sim is (N,2) render coords.
        # It's easier to just disable wireframe or update it correctly.
        # Let's Skip updating self.pts for now as wireframe is debug.


if __name__ == "__main__":
    # 載入設定檔
    # game = Live2DEngine(tracker=FakeTracker())
    # game = Live2DEngine(tracker=AsyncFaceTracker())
    game = TestMesh()
    arcade.run()
