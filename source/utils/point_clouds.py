"""
Util functions for working with point clouds.
"""

from __future__ import annotations

import copy
import os

import numpy as np

import open3d as o3d
from utils import recursive_config
from utils.coordinates import Pose3D, get_circle_points
from utils.importer import PointCloud
from utils.mask3D_interface import get_coordinates_from_item


def add_coordinate_system(
    cloud: PointCloud,
    color: (int, int, int),
    ground_coordinate: np.ndarray | None = None,
    e1: np.ndarray | None = None,
    e2: np.ndarray | None = None,
    e3: np.ndarray | None = None,
    e_relative_to_ground: bool = True,
    size: int = 1,
) -> PointCloud:
    """
    Given a point cloud, add a coordinate system to it.
    It adds three axes, where x has the lowest distance between spheres, y the second lowest, and z the highest.
    :param cloud: original point cloud
    :param color: color of the coordinate system to add
    :param ground_coordinate: center of the coordinate system
    :param e1: end of x-axis
    :param e2: end of y-axis
    :param e3: end of z-axis
    :param e_relative_to_ground: whether e is specified relative to the ground coordinate or not
    :param size: size multiplier for axes
    :return: point cloud including new axes
    """
    nrx, nry, nrz = 40 * size, 20 * size, 5 * size
    if ground_coordinate is None:
        ground_coordinate = np.asarray([0, 0, 0])
    if e1 is None:
        e1 = np.asarray([1, 0, 0])
    if e2 is None:
        e2 = np.asarray([0, 1, 0])
    if e3 is None:
        e3 = np.asarray([0, 0, 1])

    if not e_relative_to_ground:
        e1 = e1 - ground_coordinate
        e2 = e2 - ground_coordinate
        e3 = e3 - ground_coordinate

    e1 = e1 * size
    e2 = e2 * size
    e3 = e3 * size

    # x
    x_vector = np.linspace(0, 1, nrx).reshape((nrx, 1))
    full_x_vector = x_vector * np.tile(e1, (nrx, 1))
    # y
    y_vector = np.linspace(0, 1, nry).reshape((nry, 1))
    full_y_vector = y_vector * np.tile(e2, (nry, 1))
    # z
    z_vector = np.linspace(0, 1, nrz).reshape((nrz, 1))
    full_z_vector = z_vector * np.tile(e3, (nrz, 1))

    full_vector = np.vstack([full_x_vector, full_y_vector, full_z_vector])
    ground_coordinate = np.tile(ground_coordinate, (full_vector.shape[0], 1))
    full_vector = full_vector + ground_coordinate

    color_vector = np.asarray(color)
    color_vector = np.tile(color_vector, (full_vector.shape[0], 1))
    return _add_points_to_cloud(cloud, full_vector, color_vector)


def _add_points_to_cloud(
    cloud: PointCloud,
    point_coordinates: np.ndarray,
    point_colors: np.ndarray,
) -> PointCloud:
    """
    Helper function to add points to a point cloud
    :param cloud: point cloud to add points to
    :param point_coordinates: coordinates of points; shape (N, 3)
    :param point_colors: colors of points; shape (N, 3)
    """
    points = np.asarray(cloud.points)
    new_points = np.vstack([points, point_coordinates])
    new_cloud_points = o3d.utility.Vector3dVector(new_points)
    cloud.points = new_cloud_points

    colors = np.asarray(cloud.colors)
    new_colors = np.vstack([colors, point_colors])
    new_cloud_colors = o3d.utility.Vector3dVector(new_colors)
    cloud.colors = new_cloud_colors
    return cloud


def body_planning(
    env_cloud: PointCloud,
    target: Pose3D,
    floor_height_thresh: float = -0.1,
    body_height: float = 0.45,
    min_distance: float = 0.75,
    max_distance: float = 1,
    lam: float = 0.5,
    n_best: int = 1,
    vis_block: bool = False,
) -> list[Pose3D]:
    """
    Plans a position for the robot to go to given a cloud *without* the item to be
    grasped, as well as point to be grasped.
    :param env_cloud: the point cloud *without* the item
    :param target: target coordinates for grasping
    :param floor_height_thresh: z value under which to cut floor
    :param body_height: height of robot body
    :param min_distance: minimum distance from object
    :param max_distance: max distance from object
    :param lam: trade-off between distance to obstacles and distance to target, higher
    lam, more emphasis on distance to target
    :param n_best: number of positions to return
    :param vis_block: whether to visualize the position
    :return: list of viable coordinates ranked by score
    """
    target = target.as_ndarray()

    # delete floor from point cloud, so it doesn't interfere with the SDF
    points = np.asarray(env_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > floor_height_thresh
    index = np.where(points_bool)[0]
    pc_no_ground = env_cloud.select_by_index(index)

    # get points radiating outwards from target coordinate
    circle_points = get_circle_points(
        resolution=64,
        nr_circles=3,
        start_radius=min_distance,
        end_radius=max_distance,
        return_cartesian=True,
    )
    ## get center of radiating circle
    target_at_body_height = target.copy()
    target_at_body_height[-1] = body_height
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    ## add the radiating circle center to the points to elevate them
    circle_points = circle_points + target_at_body_height
    ## filter every point that is outside the scanned scene
    circle_points_bool = (min_points <= circle_points) & (circle_points <= max_points)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool]
    filtered_circle_points = filtered_circle_points.reshape((-1, 3))

    # transform point cloud to mesh to calculate SDF from
    ball_sizes = (0.02, 0.011, 0.005)
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    mesh_no_ground = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc_no_ground, radii=ball_sizes
    )
    mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_no_ground_legacy)

    # of the filtered points, cast ray from target to point to see if there are
    # collisions
    ray_directions = filtered_circle_points - target
    rays_starts = np.tile(target, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 3
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    circle_tensors = o3d.core.Tensor(
        filtered_circle_points, dtype=o3d.core.Dtype.Float32
    )

    # calculate the best body positions
    ## calculate SDF distances
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    ## calculate distance to target
    target = target.reshape((1, 1, 3))
    target_distances = filtered_circle_points - target
    target_distances = target_distances.squeeze()
    target_distances = np.linalg.norm(target_distances, ord=2, axis=-1)
    ## get the top n coordinates
    values_to_maximize = distances - lam * target_distances

    # Flatten the array and get the indices that would sort it in descending order
    flat_indices = np.argsort(-values_to_maximize.flatten())

    # Get the indices of the top n entries
    top_n_indices = np.unravel_index(flat_indices[:n_best], values_to_maximize.shape)

    # Get the corresponding values
    top_n_coordinates = filtered_circle_points[top_n_indices]

    if vis_block:
        # draw the entries in the cloud
        x = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        x.translate(target.reshape(3, 1))
        x.paint_uniform_color((1, 0, 0))
        y = copy.deepcopy(env_cloud)
        y = add_coordinate_system(y, (1, 0, 0), (0, 0, 0))
        drawable_geometries = [x, y]
        for idx, coordinate in enumerate(top_n_coordinates, 1):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(coordinate)
            color = np.asarray([0, 1, 0]) * (idx / (2 * n_best) + 0.5)
            sphere.paint_uniform_color(color)
            drawable_geometries.append(sphere)
        o3d.visualization.draw_geometries(drawable_geometries)

    poses = []
    for coord in top_n_coordinates:
        pose = Pose3D(coord)
        pose.set_rot_from_direction(target - coord)
        poses.append(pose)
    return poses


def get_radius_env_cloud(
    item_cloud: PointCloud, env_cloud: PointCloud, radius: float
) -> PointCloud:
    """
    Given two point clouds, one representing an item, one representing its environment, extract all points from the
    environment cloud that are within a certain radius of the item center.
    :param item_cloud: point cloud of item
    :param env_cloud: point cloud of environment
    :param radius: radius in which to extract points
    """
    center = np.mean(np.asarray(item_cloud.points), axis=0)
    distances = np.linalg.norm(np.asarray(env_cloud.points) - center, ord=2, axis=1)
    env_cloud = env_cloud.select_by_index(np.where(distances < radius)[0].tolist())
    return env_cloud


def icp(
    pcd1: PointCloud,
    pcd2: PointCloud,
    threshold: float = 0.2,
    trans_init: np.ndarray | None = None,
    max_iteration: int = 5000,
    point_to_point: bool = False,
) -> np.ndarray:
    """
    Return pcd1_tform_pcd2 via ICP.
    :param pcd1: First point cloud
    :param pcd2: Second point cloud
    :param threshold: threshold for alignment in ICP
    :param trans_init: initial transformation matrix guess (default I_4)
    :param max_iteration: maximum iterations for ICP
    :param point_to_point: whether to use Point2Point (true) or Point2Plane (false) ICP
    :return: pcd1_tform_pcd2
    """
    if trans_init is None:
        trans_init = np.eye(4)

    if point_to_point:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration
    )

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        threshold,
        init=trans_init,
        estimation_method=method,
        criteria=criteria,
    )
    return reg_p2p.transformation


def test():
    ITEM, INDEX = "potted plant", 0
    config = recursive_config.Config()
    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")

    item_cloud, environment_cloud = get_coordinates_from_item(
        ITEM, mask_path, pc_path, INDEX
    )
    x = copy.deepcopy(item_cloud)
    x.paint_uniform_color((1, 0, 0))
    y = copy.deepcopy(environment_cloud)
    y = add_coordinate_system(y, (1, 1, 1), (0, 0, 0))

    end_coordinates = np.mean(np.asarray(item_cloud.points), axis=0)
    end_coordinates = Pose3D(end_coordinates)
    print(end_coordinates)
    robot_target = body_planning(
        environment_cloud,
        end_coordinates,
        min_distance=0.6,
        max_distance=1,
        n_best=10,
        vis_block=True,
    )[0]
    print(robot_target)

    o3d.visualization.draw_geometries([x, y])


if __name__ == "__main__":
    test()
