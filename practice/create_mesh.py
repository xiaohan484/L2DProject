import numpy as np
import cv2
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import triangle as tr
from shapely.geometry import LineString
from shapely.ops import unary_union

# Get the absolute path of the directory one level up
# __file__ is the path to the current script
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
from Const import MODEL_PATH


def create_image_mesh(image_path, debug=True):
    # 1. Load image and convert to grayscale
    # We use OpenCV to detect edges
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = img[:, :, 3] == 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img[mask] = [255, 255, 255, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray[mask] = 255
    ret, binary_img = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 4. 繪製結果 (Optional)
    max_contour = max(contours, key=cv2.contourArea)
    # Triangulate
    # 'p': Planar Straight Line Graph (Constrained Delaunay)
    # 'q30': Quality mesh (no angle less than 30 degrees)
    # 'a500': Max area of triangles (adjust this to change density)
    # 'D': Conforming Delaunay
    vertices = max_contour.reshape(-1, 2).astype(np.float64)
    n = len(vertices)
    segments = np.array([[i, (i + 1) % n] for i in range(n)])
    input_data = {"vertices": vertices, "segments": segments}
    mesh = tr.triangulate(input_data, "pq30a100")
    pts = mesh["vertices"]
    tris = mesh["triangles"]
    # Plot the mesh
    if debug is True:
        plt.imshow(img)
        plt.triplot(pts[:, 0], pts[:, 1], tris, color=(0, 1, 0, 0.5), linewidth=0.5)
        plt.title("Constrained Delaunay Mesh on Hair")
        plt.axis("off")
        plt.show()
    return


# Execute the function
create_image_mesh(f"{MODEL_PATH}/FrontHairLeft.png")
