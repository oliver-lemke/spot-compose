from __future__ import annotations

import os

import numpy as np

import open3d as o3d
from utils import recursive_config
from utils.coordinates import Coordinates, grasp_from_direction
from utils.mask3D_interface import get_coordinates_from_item


def get_gripper_bounding_box(
    target_coordinates: Coordinates,
    direction: Coordinates,
    distance: float = 0.0,
    box_dims: (float, float, float) = (0.5, 0.15, 0.15),
    # box_dims: (float, float, float) = (0.25, 0.15, 0.15),
):
    # adjust distance so that it is measured from the x_start of the bounding box
    distance = distance + box_dims[0] / 2

    # get coordinates of box rotated and translated
    xyz = target_coordinates.get_as_ndarray()
    adjusted_coordinates = grasp_from_direction(
        direction, distance=distance, roll=0, use_degrees=False
    )
    center = adjusted_coordinates.get_as_ndarray() + xyz
    rot_matrix = adjusted_coordinates.rot.to_matrix()

    # get bounding box
    extents = np.asarray(box_dims)
    bounding_box = o3d.geometry.OrientedBoundingBox(center, rot_matrix, extents)
    return bounding_box


def main() -> None:
    # CONSTANTS
    POINT4 = (-0.9261271315789458, 0.7401371907894729, 0.1)
    TARGET_COORDINATES = Coordinates(POINT4)
    ITEM = "backpack"
    DIRECTION = Coordinates((1, 0, 0))

    # paths
    config = recursive_config.Config()
    directory_path = config.get_subpath("aligned_point_clouds")
    pc_path = os.path.join(
        str(directory_path), config["pre_scanned_graphs"]["high_res"]
    )
    pc_path = os.path.join(pc_path, "scene.ply")

    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    # coordinates from item
    item_cloud, environment_cloud = get_coordinates_from_item(ITEM, mask_path, pc_path)
    end_coordinates = np.mean(np.asarray(item_cloud.points), axis=0)
    TARGET_COORDINATES = Coordinates(end_coordinates)

    # get item and env meshes
    ball_sizes = (0.02, 0.011, 0.005)
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    item_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        item_cloud, radii=ball_sizes
    )
    environment_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        environment_cloud, radii=ball_sizes
    )
    item_mesh.paint_uniform_color(np.asarray([1, 0, 1]))

    # get bounding box for gripper
    gripper_bounding_box = get_gripper_bounding_box(
        TARGET_COORDINATES,
        DIRECTION,
    )
    gripper_bounding_box = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
        gripper_bounding_box
    )

    intersect = environment_mesh.is_intersecting(gripper_bounding_box)
    if intersect:
        color = np.asarray([1, 0, 0])
    else:
        color = np.asarray([0, 1, 0])
    gripper_bounding_box.paint_uniform_color(color)
    o3d.visualization.draw_geometries(
        [environment_mesh + item_mesh + gripper_bounding_box]
    )


if __name__ == "__main__":
    main()
