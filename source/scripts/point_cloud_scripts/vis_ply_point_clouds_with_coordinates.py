# https://stackoverflow.com/a/52226098/19032494
from __future__ import annotations

import os

import numpy as np

import open3d as o3d
from scipy.spatial.transform import Rotation
from utils import recursive_config
from utils.coordinates import Pose3D, pose_distanced
from utils.importer import PointCloud
from utils.point_clouds import add_coordinate_system


def draw_cloud(cloud: PointCloud):
    o3d.visualization.draw_geometries([cloud])  # Visualize point cloud


def load_points(path: str) -> PointCloud:
    cloud = o3d.io.read_point_cloud(path)  # Read point cloud
    return cloud


def rotate_e(rpy: (float, float, float)) -> (np.ndarray, np.ndarray, np.ndarray):
    rot_matrix = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    e1, e2, e3 = rot_matrix.T
    return e1, e2, e3


def main():
    POINT1 = (0.29, -0.48, 0.8)
    RPY1 = (0, 0, 180)
    KNOB_COORDINATES_TOP = (0.34, -1.58, 0.52)
    KNOB_COORDINATES_BOT = (0.34, -1.58, 0.32)
    BUTTON_COORDINATES = (0.64, -1.55, 0.5)
    RPYPUSH = (0, 90, 120)
    STAND_COORDINATES = (0.45, -0.93, 1.05)
    RPYSEARCH = (0, 0, 180)
    option = 0

    config = recursive_config.Config()
    if option == 0:  # aligned
        dir_path = config.get_subpath("aligned_point_clouds")
        path = os.path.join(
            str(dir_path), config["pre_scanned_graphs"]["high_res"], "scene.ply"
        )
    elif option == 1:  # high-res
        dir_path = config.get_subpath("prescans")
        path = os.path.join(
            str(dir_path),
            f'{config["pre_scanned_graphs"]["high_res"]}',
            "point_cloud.ply",
        )
    else:  # low-res
        dir_path = config.get_subpath("point_clouds")
        path = os.path.join(
            str(dir_path), f'{config["pre_scanned_graphs"]["low_res"]}.ply'
        )

    print(path)
    cloud = load_points(path)
    ground = np.asarray((0, 0, 0))
    cloud = add_coordinate_system(cloud, (0, 255, 0), ground, size=2)
    es1 = rotate_e(RPY1)
    cloud = add_coordinate_system(cloud, (255, 0, 255), POINT1, *es1, size=2)
    es2 = rotate_e(RPYSEARCH)
    es3 = rotate_e(RPYPUSH)
    cloud = add_coordinate_system(
        cloud, (255, 0, 255), KNOB_COORDINATES_TOP, *es2, size=2
    )
    cloud = add_coordinate_system(
        cloud, (255, 0, 255), KNOB_COORDINATES_BOT, *es2, size=2
    )
    cloud = add_coordinate_system(
        cloud, (255, 0, 255), BUTTON_COORDINATES, *es3, size=2
    )
    cloud = add_coordinate_system(cloud, (255, 0, 255), STAND_COORDINATES, *es2, size=2)

    draw_cloud(cloud)


if __name__ == "__main__":
    main()
