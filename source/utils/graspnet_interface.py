"""
Util functions for communicating with graspnet docker server.
"""

from __future__ import annotations

import copy
import heapq
import os
from logging import Logger
from typing import Any, Optional

import numpy as np

import open3d as o3d
from utils import recursive_config
from utils.coordinates import (
    Pose3D,
    _rotation_from_direction,
    get_uniform_sphere_directions,
    remove_duplicate_rows,
)
from utils.docker_communication import save_files, send_request
from utils.files import prep_tmp_path
from utils.importer import PointCloud
from utils.openmask_interface import get_mask_points
from utils.point_clouds import get_radius_env_cloud
from utils.recursive_config import Config

# CONSTANTS
# max gripper width is 0.175m, but in nn is 0.100m, therefore we scale models
SCALE = 0.1 / 0.175
# MAX_GRIPPER_WIDTH = 0.06
MAX_GRIPPER_WIDTH = 0.07
GRIPPER_HEIGHT = 0.24227 * SCALE


def _get_rotation_matrices(resolution: int) -> np.ndarray:
    """
    The network only predicts grasps from the current perspective. To get more
    possibilities, we have to look at the object from a lot of different angles.
    To do this, we specify a bunch of rotation matrices to rotate the object.
    :param resolution: how many angles angular degree
    :return: rotation matrices of shape (nr_angles, 3, 3)
    """
    # first get some uniform directions
    directions = get_uniform_sphere_directions(resolution)
    directions = directions.reshape((-1, 3))
    # in this case we don't want duplicate rows
    directions = remove_duplicate_rows(directions, tolerance=1e-5)
    rot_matrices = []
    invariant_direction = (0, 0, 1)
    # convert from directions to rotation matrices to look at the object from there
    for direction in directions:
        direction = tuple(direction.tolist())
        rot_matrix = _rotation_from_direction(
            direction,
            roll=0,
            invariant_direction=invariant_direction,
            degrees=False,
            invert=True,
        )
        rot_matrices.append(rot_matrix)
    rot_matrices = np.stack(rot_matrices)
    return rot_matrices


def _filter(
    contents: dict, item_cloud: PointCloud, limits: np.ndarray, thresh: float = 0.02
) -> list[(int, int)]:
    """
    Filter for all grasps that have a positive score (i.e. exist), and are on the item point cloud.
    :param contents: contents of the request
    :param item_cloud: point cloud of the item to grasp
    :param limits: limit of the grasp position
    :param thresh: maximum distance of a grasp from the nearest point in the cloud
    :return: list of indices with grasps that fulfill condition
    """
    scoress = contents["scoress"]
    tf_matricess = contents["tf_matricess"]

    points = np.asarray(item_cloud.points)

    mins = limits[0]
    maxs = limits[1]
    center = np.asarray([0, 0, 0, 1])
    indices = []
    for idx_rot, tf_matrices in enumerate(tf_matricess):
        for idx_nr, tf_matrix in enumerate(tf_matrices):
            if scoress[idx_rot, idx_nr] == -1:
                continue
            grip_point = tf_matrix @ center
            grip_point = grip_point[:3] / grip_point[3]
            outside = np.any((grip_point < mins) | (grip_point > maxs))
            if outside:
                continue
            grip_point = np.expand_dims(grip_point, 0)
            distances = np.linalg.norm(points - grip_point, axis=1, ord=2)
            min_distance = np.min(distances, axis=0)
            if min_distance >= thresh:
                continue
            indices.append((idx_rot, idx_nr))
    return indices


def get_k_best_scores(scores_dict: dict[Any, float], k: int) -> list[tuple[Any, float]]:
    """
    Returns the k elements with the highest scores from scores_dict.

    Args:
    scores_dict (Dict[Any, float]): A dictionary with elements as keys and their scores as values.
    k (int): The number of top elements to return.

    Returns:
    List[Tuple[Any, float]]: A list of tuples containing the k elements with the highest scores and their corresponding scores.
    """
    # Use heapq.nlargest to find the k elements with the highest scores
    # It requires an iterable and a function (lambda x: x[1]) to extract comparison keys (scores in this case)
    k_best = heapq.nlargest(k, scores_dict.items(), key=lambda x: x[1])

    return k_best


def _predict(
    item_cloud: PointCloud,
    env_cloud: PointCloud,
    limits: np.ndarray,
    rotations: np.ndarray,
    config: Config,
    logger: Optional[Logger] = None,
    top_n: int = 3,
    n_best: int = 1,
    timeout: int = 90,
    top_down_grasp: bool = False,
    vis_block: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Predict grasps using graspnet
    :param item_cloud: the point cloud of the item we want to grasp
    :param env_cloud: the point cloud of the environment (for grasp filtering)
    :param limits: spatial limits within which to search for gras, shape (2, 3) with rows being minimum and maximum
    x,y,z respectively
    :param rotations: rotations of the scene (to predict grasps from different angles)
    :param config:
    :param logger:
    :param top_n: per viewing angle, how many grasps to predict
    :param timeout: timeout for the server request
    :param top_down_grasp: allow for top-down grasps (not 100% sure about this, no documentation in graspnet)
    :param vis_block: whether to block execution for intermediate visualization
    """
    assert limits.shape == (2, 3)
    assert rotations.shape[-2:] == (3, 3)

    kwargs = {
        "max_gripper_width": ("float", MAX_GRIPPER_WIDTH),
        "gripper_height": ("float", GRIPPER_HEIGHT),
        "top_n": ("int", top_n),
        "vis": ("bool", vis_block),
        "top_down_grasp": ("bool", top_down_grasp),
    }

    address_details = config["servers"]["graspnet"]
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    tmp_path = prep_tmp_path(config)

    center = np.zeros((3, 1))
    item_cloud = copy.deepcopy(item_cloud).scale(SCALE, center)
    env_cloud = copy.deepcopy(env_cloud).scale(SCALE, center)
    limits = limits.copy() * SCALE

    pcd = item_cloud + env_cloud
    save_data = [
        ("points.npy", np.save, np.asarray(pcd.points)),
        ("colors.npy", np.save, np.asarray(pcd.colors)),
        ("limits.npy", np.save, limits),
        ("rotations.npy", np.save, rotations),
    ]
    points_path, colors_path, limits_path, rotations_path = save_files(
        save_data, tmp_path
    )

    paths_dict = {
        "points": points_path,
        "colors": colors_path,
        "limits": limits_path,
        "rotations": rotations_path,
    }
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, kwargs, timeout, tmp_path)
    if logger:
        logger.info(f"Received response!")

    # get gripper meshes (already transformed)
    tf_matricess = contents["tf_matricess"]
    scoress = contents["scoress"]
    widthss = contents["widthss"]

    # get the indices of the valid grasp positions (viewing angle, nr)
    good_indices = _filter(contents, item_cloud, limits)
    # select corresponding transformation matrices, scores, and widths
    tf_matrices_dict = {(rot, nr): tf_matricess[rot, nr] for (rot, nr) in good_indices}
    scores_dict = {(rot, nr): scoress[rot, nr] for (rot, nr) in good_indices}
    widths_dict = {(rot, nr): widthss[rot, nr] for (rot, nr) in good_indices}

    maxes = get_k_best_scores(scores_dict, k=n_best)

    if vis_block:
        # ball_sizes = np.asarray((0.02, 0.011, 0.005)) * SCALE
        # ball_sizes = o3d.utility.DoubleVector(ball_sizes)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     pcd, radii=ball_sizes
        # )
        grippers = {
            (rot, nr): contents[f"mesh_{rot:03}_{nr:03}"] for rot, nr in good_indices
        }
        # visualize all best per angle
        o3d.visualization.draw_geometries([pcd, *grippers.values()])
        # visualize best gripper only
        grippers_display = [grippers[argmax] for (argmax, _) in maxes]
        o3d.visualization.draw_geometries([pcd, *grippers_display])
        o3d.visualization.draw_geometries([pcd, grippers_display[0]])

    tf_returns = []
    width_returns = []
    score_returns = []
    for argmax, score in maxes:
        tf = tf_matrices_dict[argmax]
        tf[:3, 3] = tf[:3, 3] / SCALE
        width = widths_dict[argmax] / SCALE
        tf_returns.append(tf)
        width_returns.append(width)
        score_returns.append(score)
    tf_returns = np.stack(tf_returns, axis=0)
    width_returns = np.stack(width_returns, axis=0)
    score_returns = np.stack(score_returns, axis=0)

    return tf_returns, width_returns, score_returns


def predict_full_grasp(
    item_cloud: PointCloud,
    env_cloud: PointCloud,
    config: recursive_config.Config,
    logger: Optional[Logger] = None,
    rotation_resolution: int = 24,
    top_n: int = 3,
    n_best: int = 1,
    timeout: int = 90,
    vis_block: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Predict a grasp position from the item point cloud and its environment.
    :param item_cloud: the point cloud of the item to be grasped
    :param env_cloud: the point cloud of the environment (typically within some radius)
    :param config: config
    :param logger: for logging
    :param rotation_resolution: number of different angles for grasp detection
    :param top_n: number of different grasps per angle
    :param timeout: seconds for http request timeout
    :param vis_block: visualize grasp before returning
    :return: transformation matrix and width of best found grasp
    """
    rotations = _get_rotation_matrices(rotation_resolution)

    item_points = np.asarray(item_cloud.points)
    mins, maxs = np.min(item_points, axis=0), np.max(item_points, axis=0)
    limits = np.stack([mins, maxs], axis=0)

    tf_matrices, widths, scores = _predict(
        item_cloud,
        env_cloud,
        limits,
        rotations,
        config,
        logger,
        top_n,
        n_best,
        timeout,
        top_down_grasp=True,
        vis_block=vis_block,
    )

    return tf_matrices, widths, scores


def predict_partial_grasp(
    pcd: PointCloud,
    original_point: Pose3D,
    tolerance: float,
    config: recursive_config.Config,
    logger: Optional[Logger] = None,
    top_n: int = 5,
    timeout: int = 90,
    vis_block: bool = False,
) -> (np.ndarray, float):
    """
    Predict a grasp position from the item point cloud and its environment.
    :param pcd: the point cloud of the item to be grasped, coordinate system must be
    that of the pose from which we view the point cloud
    :param original_point: original grasping point
    :param tolerance: expected maximum drift in m
    :param config: config
    :param logger: for logging
    :param top_n: number of different grasps per angle
    :param timeout: seconds for http request timeout
    :param vis_block: visualize grasp before returning
    :return: transformation matrix and width of best found grasp
    """
    mins = original_point.as_ndarray() - tolerance
    maxs = original_point.as_ndarray() + tolerance
    limits = np.stack((mins, maxs), axis=0)
    rotations = np.eye(3).reshape(1, 3, 3)

    points = np.asarray(pcd.points)
    inside_points = np.all(mins <= points, axis=1) & np.all(points <= maxs, axis=1)
    inside_points_idx = np.where(inside_points)[0].tolist()

    # delete some space around the item_cloud to facilitate finding grasps
    frame_width = 0.03
    mins_ext = mins - frame_width
    maxs_ext = maxs + frame_width
    inside_ext_points = np.all(mins_ext <= points, axis=1) & np.all(
        points <= maxs_ext, axis=1
    )
    inside_ext_points_idx = np.where(inside_ext_points)[0].tolist()

    item_cloud = pcd.select_by_index(inside_points_idx)
    env_cloud = pcd.select_by_index(inside_ext_points_idx, invert=True)

    if vis_block:
        x = copy.deepcopy(item_cloud).paint_uniform_color((1, 0, 0))
        y = copy.deepcopy(env_cloud).paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([x, y])

    tf_matrix, width = _predict(
        item_cloud,
        env_cloud,
        limits,
        rotations,
        config,
        logger,
        top_n,
        timeout,
        False,
        vis_block=vis_block,
    )

    return tf_matrix, width


########################################################################################
########################################################################################
####################################### TESTING ########################################
########################################################################################
########################################################################################


def _test_full_grasp() -> None:
    config = Config()
    ITEM, INDEX = "green watering can", 0
    RADIUS = 0.75
    RES = 16
    VIS_BLOCK = True

    item_cloud, environment_cloud = get_mask_points(
        ITEM, config, idx=INDEX, vis_block=True
    )
    if VIS_BLOCK:
        o3d.visualization.draw_geometries([item_cloud])
    lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, RADIUS)

    ###############################################################################
    ################################ ARM COMMANDS #################################
    ###############################################################################

    # robot_target = body_planning(environment_cloud, end_coordinates, vis_block=True)[0]
    print("Request")
    tf_matrices, _, scores = predict_full_grasp(
        item_cloud,
        lim_env_cloud,
        config,
        rotation_resolution=RES,
        top_n=5,
        n_best=20,
        vis_block=VIS_BLOCK,
    )
    print(tf_matrices)


def _test_limited_grasp() -> None:
    IMAGE_NR = 119
    DIMS = 1920, 1440
    TOLERANCE = 0.1

    config = Config()
    base_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    d_path = os.path.join(str(base_path), ending, "depth", f"{IMAGE_NR}.png")
    e_path = os.path.join(str(base_path), ending, "pose", f"{IMAGE_NR}.txt")

    f, cx, cy = 1439.00, 966.02, 722.89
    ints = o3d.camera.PinholeCameraIntrinsic(*DIMS, f, f, cx, cy)
    exts = np.loadtxt(e_path)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.io.read_image(d_path),
        ints,
        np.eye(4),
        depth_scale=1000,
    )

    grip_pose = Pose3D((0, 0, 0.45))

    # sphere = TriangleMesh.create_sphere(radius=0.05)
    # grip = copy.deepcopy(sphere).translate(grip_pose.as_ndarray())
    # o3d.visualization.draw_geometries([pcd, sphere, grip])

    tf_matrix, width = predict_partial_grasp(
        pcd, grip_pose, TOLERANCE, config, None, vis_block=True
    )
    print(tf_matrix, width)


if __name__ == "__main__":
    _test_full_grasp()
    # _test_limited_grasp()
