"""
Utils for visualization.
"""

import copy

import numpy as np

import cv2
import open3d as o3d
from utils.importer import PointCloud, Vector3dVector


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image prior to plotting.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def show_image(image: np.ndarray, title: str = "Image"):
    """
    Show an RGB image.
    """
    normalized_image = normalize_image(image)
    cv2.imshow(title, normalized_image)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key is pressed
            break
    cv2.destroyAllWindows()
    return normalized_image


def show_depth_image(depth_image: np.ndarray, title: str = "Depth Image"):
    """
    Show a colormapped depth image.
    """
    min_val = np.min(depth_image)
    max_val = np.max(depth_image)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (depth_image - min_val)).astype("uint8")
    depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    colored_image = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    return show_image(colored_image, title)


def show_two_geometries_colored(
    geometry1, geometry2, color1=(1, 0, 0), color2=(0, 1, 0)
) -> None:
    """
    Given two open3d geometries, color them and visualize them.
    :param geometry1:
    :param geometry2:
    :param color1: color for first geometry
    :param color2: color for second geometry
    """
    geometry1_colored = copy.deepcopy(geometry1)
    geometry2_colored = copy.deepcopy(geometry2)
    geometry1_colored.paint_uniform_color(color1)
    geometry2_colored.paint_uniform_color(color2)
    o3d.visualization.draw_geometries([geometry1_colored, geometry2_colored])


def show_point_cloud_in_out(points: np.ndarray, in_mask: np.ndarray) -> None:
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    pcd_in = pcd.select_by_index(np.where(in_mask)[0])
    pcd_out = pcd.select_by_index(np.where(~in_mask)[0])
    show_two_geometries_colored(pcd_out, pcd_in)
