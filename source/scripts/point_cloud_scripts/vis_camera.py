# https://stackoverflow.com/a/52226098/19032494
from __future__ import annotations

import json
import os

import numpy as np

import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from utils.point_clouds import add_coordinate_system
from utils.recursive_config import Config


def rotate_e(
    rpy: (float, float, float), degrees: bool = True
) -> (np.ndarray, np.ndarray, np.ndarray):
    rot_matrix = Rotation.from_euler("xyz", rpy, degrees=degrees).as_matrix()
    e1, e2, e3 = rot_matrix.T
    return e1, e2, e3


def main():
    # set up paths
    config = Config()
    data_path = config.get_subpath("data")
    base_path = os.path.join(data_path, "24-02-22_15_24_26", "2024_02_22_15_17_21")
    frame_nr = 147
    frame_nr_str = str(frame_nr).zfill(5)
    json_name = f"frame_{frame_nr_str}.json"
    json_path = os.path.join(base_path, json_name)
    mesh_path = os.path.join(base_path, "textured_output.obj")

    with open(json_path, "r") as file:
        camera_dict = json.load(file)
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=50_000, use_triangle_normal=True
    )

    extrinsics = np.array(camera_dict["cameraPoseARFrame"]).reshape((4, 4))
    correct_matrix = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    extrinsics[:3, :3] = extrinsics[:3, :3] @ correct_matrix

    coords = extrinsics[:3, 3]
    rpy = Rotation.from_matrix(extrinsics[:3, :3]).as_euler("xyz", degrees=True)

    ground = np.asarray((0, 0, 0))
    geom = add_coordinate_system(pcd, (0, 255, 0), ground, size=2)
    geom = add_coordinate_system(geom, (255, 0, 255), coords, *rotate_e(rpy), size=2)

    o3d.visualization.draw_geometries([geom])


if __name__ == "__main__":
    main()
