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
    KNOB_COORDINATES_TOP = (0.62, -1.34, -0.1)
    RPY_KNOB = (0, 0, 0)
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
    es_knob = rotate_e(RPY_KNOB)
    cloud = add_coordinate_system(
        cloud, (255, 0, 255), KNOB_COORDINATES_TOP, *es_knob, size=2
    )

    draw_cloud(cloud)


if __name__ == "__main__":
    main()
