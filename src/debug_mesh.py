import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.create_mesh import create_image_mesh
from src.Const import MODEL_PATH


def test_density_and_simplify():
    image_path = os.path.join(MODEL_PATH, "FrontHairLeft.png")
    print(f"Testing simplification for: {image_path}")

    # Load image and get contour manually to test approxPolyDP impact
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = img[:, :, 3] == 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img[mask] = [255, 255, 255, 255]
    gray[mask] = 255
    ret, binary_img = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = max(contours, key=cv2.contourArea)

    original_points = len(max_contour)
    print(f"Original contour vertices: {original_points}")

    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for eps in epsilons:
        approx = cv2.approxPolyDP(max_contour, eps, True)
        print(f"Epsilon {eps}: {len(approx)} vertices")


if __name__ == "__main__":
    test_density_and_simplify()
