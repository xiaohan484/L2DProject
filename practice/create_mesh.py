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

debug = True


def remove_duplicates_preserve_order(pts):
    seen = set()
    cleaned = []
    for p in pts:
        t_p = tuple(p)
        if t_p not in seen:
            print(t_p)
            cleaned.append(p)
            seen.add(t_p)
        else:
            print("seen:", t_p)
    return np.array(cleaned, dtype=np.float64)


def safe_triangulate(coords):
    # 1. Create a LineString from the coordinates
    # coords: [(x1, y1), (x2, y2), ...]
    line = LineString(coords + coords[0])  # Close the loop
    # 2. Use unary_union to split segments at all intersection points
    # This turns a self-intersecting line into a collection of valid segments
    cleaned_lines = unary_union(line)

    # 3. Extract vertices and segments for the triangle library
    nodes = []
    segments = []

    # Map vertices to indices to build the segments list
    node_map = {}

    def get_node_idx(pt):
        if pt not in node_map:
            node_map[pt] = len(nodes)
            nodes.append(pt)
        return node_map[pt]

    if cleaned_lines.geom_type == "MultiLineString":
        print("multiline")
        for l in cleaned_lines.geoms:
            x, y = l.xy
            plt.plot(x, y)
            pts = list(l.coords)
            for i in range(len(pts) - 1):
                idx1 = get_node_idx(pts[i])
                idx2 = get_node_idx(pts[i + 1])
                segments.append([idx1, idx2])
    else:
        pts = list(cleaned_lines.coords)
        for i in range(len(pts) - 1):
            idx1 = get_node_idx(pts[i])
            idx2 = get_node_idx(pts[i + 1])
            segments.append([idx1, idx2])

    # 4. Input into triangle library
    # 'p' means PSLG input
    data = dict(vertices=np.array(nodes), segments=np.array(segments))
    return data


def create_image_mesh(image_path, num_random_points=500):
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
    plt.imshow(img)
    plt.triplot(pts[:, 0], pts[:, 1], tris, color="cyan", linewidth=0.5)
    # plt.plot(vertices[:, 0], vertices[:, 1], "o", color="red", markersize=1)

    plt.title("Constrained Delaunay Mesh on Hair")
    plt.axis("off")
    plt.show()
    return


# Execute the function
create_image_mesh(f"{MODEL_PATH}/FrontHairLeft.png")
