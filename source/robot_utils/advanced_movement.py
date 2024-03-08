from __future__ import annotations

import copy
import time

import numpy as np

import open3d as o3d
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME
from robot_utils.basic_movements import move_arm_distanced, move_body, set_gripper
from robot_utils.frame_transformer import (
    GRAPH_SEED_FRAME_NAME,
    VISUAL_SEED_FRAME_NAME,
    FrameTransformerSingleton,
)
from robot_utils.video import (
    GRIPPER_DEPTH,
    get_d_pictures,
    point_cloud_from_camera_captures,
)
from scipy.spatial.transform import Rotation
from utils import vis
from utils.coordinates import (
    Pose2D,
    Pose3D,
    from_a_to_b_distanced,
    spherical_angle_views_from_target,
)
from utils.graspnet_interface import predict_partial_grasp
from utils.importer import PointCloud
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.singletons import (
    RobotCommandClientSingleton,
    RobotSingleton,
    WorldObjectClientSingleton,
)
from utils.user_input import confirm_coordinates

frame_transformer = FrameTransformerSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
world_object_client = WorldObjectClientSingleton()


def rotate(
        end_pose: Pose2D,
        frame_name: str,
) -> None:
    """
    Rotate in position to rotation of end_pose
    :param end_pose: will rotate to match end_pose.rot_matrix
    :param frame_name: frame of the end_pose
    """
    start_pose = frame_transformer.get_current_body_position_in_frame(
        end_frame=frame_name
    )
    start_pose = Pose2D.from_bosdyn_pose(start_pose)

    # calculate rotation
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, 0)

    ###############################################################################
    ############################## MOVEMENT COMMANDS ##############################
    ###############################################################################

    mov_static_params = {
        "frame_name": frame_name,
    }

    # rotate body in position
    rotated_pose = Pose2D(start_pose.coordinates, destination_pose.rot_matrix)
    move_body(rotated_pose, **mov_static_params)


def rotate_and_move_distanced(
        end_pose: Pose2D,
        distance: float,
        frame_name: str,
        sleep: bool = True,
) -> None:
    """
    First rotate, then move to a location, with a certain distance offset.
    The distance is in the direction of the current pose.
    :param end_pose: final pose to walk towards (minus the distance)
    :param distance: distance offset to final pose
    :param frame_name: frame of the end_pose
    :param sleep: whether to sleep in between movements for safety
    """
    sleep_multiplier = 1 if sleep else 0

    start_pose = frame_transformer.get_current_body_position_in_frame(
        end_frame=frame_name
    )
    start_pose = Pose2D.from_bosdyn_pose(start_pose)

    # calculate destination pose
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)

    confirm = confirm_coordinates(start_pose, end_pose, destination_pose, distance)

    assert confirm, "Aborting due to negative confirmation!"

    ###############################################################################
    ############################## MOVEMENT COMMANDS ##############################
    ###############################################################################

    # rotate body on position
    time.sleep(1 * sleep_multiplier)
    rotate(end_pose, frame_name=frame_name)
    time.sleep(1 * sleep_multiplier)

    # walk
    move_body(destination_pose, frame_name=frame_name)

    time.sleep(1 * sleep_multiplier)


def move_body_distanced(
        end_pose: Pose2D,
        distance: float,
        frame_name: str,
        sleep: bool = True,
) -> None:
    """
    Move to a location, with a certain distance offset.
    The distance is in the direction of the current pose.
    :param end_pose: final pose to walk towards (minus the distance)
    :param distance: distance offset to final pose
    :param frame_name: frame of the end_pose
    :param sleep: whether to sleep in between movements for safety
    """
    sleep_multiplier = 1 if sleep else 0

    start_pose = frame_transformer.get_current_body_position_in_frame(
        end_frame=frame_name
    )
    start_pose = Pose2D.from_bosdyn_pose(start_pose)

    # calculate destination pose
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)

    ###############################################################################
    ############################## MOVEMENT COMMANDS ##############################
    ###############################################################################

    mov_static_params = {
        "frame_name": frame_name,
    }
    move_body(destination_pose, **mov_static_params)
    time.sleep(1 * sleep_multiplier)


def positional_grab(
        pose: Pose3D,
        distance_start: float,
        distance_end: float,
        frame_name: str,
        already_gripping: bool = False,
        **kwargs,
) -> None:
    """
    Grab something at a specified position. The gripper will first move towards the distanced pose, which is "pose",
    but offset by distance_start in the opposite direction of the viewing direction.
    Then it will move towards the specified pose offset by distance_end.
    So in essence, the direction of the pose specifies the axis along which it moves.
    :param pose: pose to grab
    :param distance_start: distance from which to start grab
    :param distance_end: distance at which to end grab
    :param frame_name: frame in which the pose is specified relative to
    :param already_gripping: whether to NOT open up the gripper in the beginning
    """
    static_params = {
        "pose": pose,
        "frame_name": frame_name,
    }

    move_arm_distanced(distance=distance_start, **static_params, **kwargs)
    set_gripper(not already_gripping)
    move_arm_distanced(distance=distance_end, **static_params, **kwargs)
    set_gripper(False)


def pull(
        pose: Pose3D,
        start_distance: float,
        mid_distance: float,
        end_distance: float,
        frame_name: str,
        stiffness_diag_in: list[int] | None = None,
        damping_diag_in: list[float] | None = None,
        stiffness_diag_out: list[int] | None = None,
        damping_diag_out: list[float] | None = None,
        forces: list[float] | None = None,
        release_after: bool = True,
        follow_arm: bool = False,
        timeout: float = 6.0,
) -> (Pose3D, Pose3D):
    """
    Executes a pulling motion (e.g. for drawers)
    :param pose: pose of knob in 3D space
    :param start_distance: how far from the knob to start grab
    :param mid_distance: how far to go before grabbing
    :param end_distance: how far to pull
    :param frame_name:
    :param release_after: release the knob after pulling motion
    :param sleep: whether to sleep in between motions for safety
    """
    assert len(stiffness_diag_in) == 6
    if stiffness_diag_in is None:
        stiffness_diag_in = [200, 500, 500, 60, 60, 60]
    assert len(damping_diag_in) == 6
    if damping_diag_in is None:
        damping_diag_in = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
    assert len(stiffness_diag_out) == 6
    if stiffness_diag_out is None:
        stiffness_diag_out = [100, 0, 0, 60, 60, 60]
    assert len(damping_diag_out) == 6
    if damping_diag_out is None:
        damping_diag_out = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
    assert len(forces) == 6
    if forces is None:
        forces = [0, 0, 0, 0, 0, 0]

    keywords_in = {
        "stiffness_diag": stiffness_diag_in,
        "damping_diag": damping_diag_in,
        "forces": forces,
        "timeout": timeout,
    }
    keywords_out = {
        "stiffness_diag": stiffness_diag_out,
        "damping_diag": damping_diag_out,
        "forces": forces,
        "timeout": timeout,
    }

    move_arm_distanced(
        pose, start_distance, frame_name, follow_arm=False
    )  # before handle
    set_gripper(True)
    move_arm_distanced(
        pose, mid_distance, frame_name, follow_arm=follow_arm, **keywords_in
    )  # moving in
    set_gripper(False)  # grab
    pull_start = frame_transformer.get_hand_position_in_frame(
        frame_name, in_common_pose=True
    )
    move_arm_distanced(
        pose, end_distance, frame_name, follow_arm=follow_arm, **keywords_out
    )  # pulling
    pull_end = frame_transformer.get_hand_position_in_frame(
        frame_name, in_common_pose=True
    )

    if release_after:
        set_gripper(True)
        move_arm_distanced(pull_end, start_distance, frame_name, follow_arm=follow_arm)

    return pull_start, pull_end


def push(
        start_pose: Pose3D,
        end_pose: Pose3D,
        start_distance: float,
        end_distance: float,
        frame_name: str,
        stiffness_diag: list[int] | None = None,
        damping_diag: list[float] | None = None,
        forces: list[float] | None = None,
        follow_arm: bool = False,
        timeout: float = 6.0,
) -> (Pose3D, Pose3D):
    """
    Executes a pushing motion (e.g. for drawers)
    :param pose: pose of knob in 3D space
    :param start_distance: how far from the button to start push
    :param end_distance: how far to push
    :param frame_name:
    :param sleep: whether to sleep in between motions for safety
    """
    assert len(stiffness_diag) == 6
    if stiffness_diag is None:
        stiffness_diag = [200, 500, 500, 60, 60, 60]
    assert len(damping_diag) == 6
    if damping_diag is None:
        damping_diag = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
    assert len(forces) == 6
    if forces is None:
        forces = [0, 0, 0, 0, 0, 0]

    keywords = {
        "stiffness_diag": stiffness_diag,
        "damping_diag": damping_diag,
        "forces": forces,
        "follow_arm": follow_arm,
        "timeout": timeout,
    }

    move_arm_distanced(
        start_pose, start_distance, frame_name, follow_arm=follow_arm
    )  # before handle
    set_gripper(False)  # fist
    move_arm_distanced(end_pose, end_distance, frame_name, **keywords)  # pushing


def adapt_grasp(body_pose: Pose3D, grasp_pose: Pose3D):
    grasp_in_body = body_pose.inverse() @ grasp_pose
    top_dir = grasp_in_body.rot_matrix @ np.array([0, 0, 1])
    to_rotate = top_dir[0] < 0

    grasp_pose_new = Pose3D(grasp_pose.coordinates.copy(), grasp_pose.rot_matrix.copy())
    if to_rotate:
        roll_matrix = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        grasp_pose_new.rot_matrix = grasp_pose_new.rot_matrix @ roll_matrix

    return grasp_pose_new


def collect_dynamic_point_cloud(
        start_pose: Pose3D,
        target_pose: Pose3D,
        frame_name: str,
        nr_captures: int = 4,
        offset: float = 10,
        degrees: bool = True,
) -> PointCloud:
    """
    Collect a point cloud of an object in front of the gripper.
    The poses from which we take the depth images are distributed spherically around the target_pose and all look at the
    target. Imagine a sphere around the target with radius of the distance between the target and the start. Now, on
    that sphere, imagine a circle around the start. The poses will lie on that circle.
    Assumes the start_pose is "looking at" the target_pose.
    :param start_pose: Pose3D describing the start position
    :param target_pose: Pose3D describing the target position (center of sphere)
    :param frame_name: frame relative to which the poses are specified
    :param nr_captures: number of poses calculated on the circle (equidistant on it)
    :param offset: offset from target to circle as an angle seen from the center
    :param degrees: whether offset is given in degrees
    :return: a point cloud stitched together from all views
    """
    if nr_captures > 0:
        angled_view_poses = spherical_angle_views_from_target(
            start_pose, target_pose, nr_captures, offset, degrees
        )
    else:
        angled_view_poses = [start_pose]
    depth_images = []
    for angled_pose in angled_view_poses:
        set_gripper(True)
        move_arm_distanced(angled_pose, 0.0, frame_name=frame_name)

        depth_image = get_d_pictures([GRIPPER_DEPTH])
        depth_images.extend(depth_image)

    pcd_odom = point_cloud_from_camera_captures(
        depth_images, frame_relative_to=ODOM_FRAME_NAME
    )

    return pcd_odom


def dynamically_refined_grasp_renew_grasp(
        pose: Pose3D,
        item_cloud: PointCloud,
        distance_start: float,
        distance_end: float,
        frame_name: str,
        nr_captures: int = 4,
        offset: float = 20,
        degrees: bool = True,
        drift_threshold: float = 0.03,
        icp_multiplier: int = 5,
        vis_block: bool = False,
) -> None:
    """
    This method implement an adaptive grasp based on collecting a new point cloud close
    to the supposed grasp position. This allows it to adjust for some drift in
    localization.
    This method specifically creates a new point cloud at the supposed position,
    calculates the transformation from original PCD to the new dynamically collected
    PCD, and transforms the grasp accordingly.
    :param pose: supposed grasp position
    :param item_cloud: cloud of the item to grab
    :param distance_start: start of the grabbing motion
    :param distance_end: end of the grabbing motion
    :param frame_name:
    :param nr_captures: number of captures to create PCD
    :param offset: increments in which to move the camera when collecting new PCD
    :param degrees: whether increments is in degrees (True) or radians (False)
    :param drift_threshold: threshold of ICP alignment
    :param icp_multiplier: how much of the point cloud around the original grasp is used for ICP alignment, in radius
    as multiple of drift_threshold
    :param vis_block: whether to visualize ICP alignment
    :return: None
    """
    capture_params = {
        "nr_captures": nr_captures,
        "offset": offset,
        "degrees": degrees,
    }

    result_pose = move_arm_distanced(pose, distance_start, frame_name=frame_name)

    # get point cloud in ODOM frame
    dynamic_pcd_odom = collect_dynamic_point_cloud(
        result_pose, pose, frame_name=frame_name, **capture_params
    )
    # transform to seed frame to match with originalF PCD
    seed_tform_odom = frame_transformer.transform_matrix(
        ODOM_FRAME_NAME, VISUAL_SEED_FRAME_NAME
    )
    # now in same frame as ordinary point cloud
    dynamic_pcd = dynamic_pcd_odom.transform(seed_tform_odom)

    original_grasp_coordinates = pose.as_ndarray()
    dynamic_points = np.array(dynamic_pcd.points)
    points_within = np.where(
        np.linalg.norm(dynamic_points - original_grasp_coordinates, axis=1)
        < icp_multiplier * drift_threshold
    )[0]
    selected_dynamic_pcd = dynamic_pcd.select_by_index(points_within)
    # voxel_size = 0.008  # for example, 0.005 unit length voxel size
    # selected_dynamic_pcd = selected_dynamic_pcd.voxel_down_sample(voxel_size)

    # perform ICP
    if not selected_dynamic_pcd.has_normals():
        selected_dynamic_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    if not item_cloud.has_normals():
        item_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    # calculate the transformation (drift) from the original cloud (seed) to the new
    # dynamically created cloud
    if vis_block:
        vis.show_two_geometries_colored(selected_dynamic_pcd, item_cloud)
    dynamic_tform_seed = icp(
        selected_dynamic_pcd,
        item_cloud,
        threshold=drift_threshold,
        max_iteration=100,
        point_to_point=True,
    )

    euler = Rotation.from_matrix(dynamic_tform_seed.copy()[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    print(euler)

    # offset original grasp pose by similar amount
    item_cloud = item_cloud.transform(dynamic_tform_seed)
    if vis_block:
        vis.show_two_geometries_colored(selected_dynamic_pcd, item_cloud)
    pose = Pose3D.from_matrix(dynamic_tform_seed) @ pose
    return positional_grab(pose, distance_start, distance_end, frame_name=frame_name)


def dynamically_refined_grasp_find_new_grasp(
        pose: Pose3D,
        distance_start: float,
        distance_end: float,
        config: Config,
        frame_name: str,
        nr_captures: int = 3,
        offset: float = 10,
        tolerance: float = 0.1,
        degrees: bool = True,
) -> None:
    """
    This method implement an adaptive grasp based on collecting a new point cloud close
    to the supposed grasp position. This allows it to adjust for some drift in
    localization.
    This method specifically creates a new point cloud at the supposed position,
    calculates the transformation from original PCD to the new dynamically collected
    PCD, and transforms the grasp accordingly.
    :param pose: supposed grasp position
    :param distance_start: start of the grabbing motion
    :param distance_end: end of the grabbing motion
    :param config:
    :param frame_name:
    :param nr_captures: number of captures to create PCD
    :param offset: increments in which to move the camera when collecting new PCD
    :param degrees: whether increments is in degrees (True) or radians (False)
    :param tolerance: tolerance for maximum possible drift
    :return: None
    """
    capture_params = {
        "nr_captures": nr_captures,
        "offset": offset,
        "degrees": degrees,
    }

    start_pose = move_arm_distanced(pose, distance_start, frame_name=frame_name)
    pcd_body = collect_dynamic_point_cloud(
        start_pose, pose, frame_name=frame_name, **capture_params
    )

    # transform into coordinate of starting pose
    seed_tform_body = frame_transformer.transform_matrix(
        BODY_FRAME_NAME, GRAPH_SEED_FRAME_NAME
    )
    pose_tform_seed = start_pose.as_matrix()
    pose_tform_body = pose_tform_seed @ seed_tform_body

    pcd_pose = pcd_body.transform(pose_tform_body)
    pose_pose = copy.deepcopy(pose)
    pose_pose.transform(pose_tform_seed)

    # get new gripper pose
    tf_matrix, _ = predict_partial_grasp(
        pcd_pose, pose_pose, tolerance, config, robot.logger
    )
    new_pose = Pose3D.from_matrix(tf_matrix)

    return positional_grab(
        new_pose, distance_start, distance_end, frame_name=frame_name
    )
