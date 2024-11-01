from __future__ import annotations

import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.advanced_movement import align_grasp_with_body, positional_grab
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_body, set_gripper, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from scipy.spatial.transform import Rotation
from utils.docker_interfaces import graspnet_interface
from utils.coordinates import Pose2D, Pose3D
from utils.logger import LoggerSingleton
from utils.docker_interfaces.openmask_interface import get_item_pcd
from utils.point_clouds import body_planning, get_radius_env_cloud
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()
logger = LoggerSingleton()


def joint_optimization_vec(
    target: Pose3D,
    tf_matrices: np.ndarray,
    widths: np.ndarray,
    scores: np.ndarray,
    body_scores: list[tuple[Pose3D, float]],
    lambda_body: float = 0.5,
    lambda_alignment: float = 1.0,
    temperature: float = 1.0,
) -> tuple[Pose3D, Pose3D, float]:
    nr_grasps = tf_matrices.shape[0]
    nr_poses = len(body_scores)
    # matrix is nr_grasps x nr_poses

    grasp_scores = scores.reshape((nr_grasps, 1))
    grasp_score_mat = np.tile(grasp_scores, (1, nr_poses))

    pose_scores = np.asarray([score for (_, score) in body_scores]).reshape(
        (1, nr_poses)
    )
    pose_score_mat = np.tile(pose_scores, (nr_grasps, 1))

    grasp_directions = [Pose3D.from_matrix(tf).direction() for tf in tf_matrices]
    grasp_directions_np = np.stack(grasp_directions, axis=0)

    body_coordinates = [pose.coordinates for (pose, _) in body_scores]
    body_coordinates_np = np.stack(body_coordinates, axis=1)
    body_to_targets_np = target.coordinates.reshape((3, 1)) - body_coordinates_np
    body_coordinates_norm = body_to_targets_np / np.linalg.norm(
        body_to_targets_np, axis=0, keepdims=True
    )

    alignment_mat = grasp_directions_np @ body_coordinates_norm
    alignment_mat_tanh = np.tanh(alignment_mat * temperature)

    joint_matrix = (
        grasp_score_mat
        + lambda_body * pose_score_mat
        + lambda_alignment * alignment_mat_tanh
    )

    argmax_index = np.argmax(joint_matrix)
    # Convert the flattened index back to a 2D index
    grasp_argmax, pose_argmax = np.unravel_index(argmax_index, joint_matrix.shape)
    print(f"{grasp_argmax=}", f"{pose_argmax=}")
    best_grasp = Pose3D.from_matrix(tf_matrices[grasp_argmax])
    width = widths[grasp_argmax]
    best_pose = body_scores[pose_argmax][0]
    return best_grasp, best_pose, width


class _BetterGrasp(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        ITEM = "candle"
        RADIUS = 0.75
        RESOLUTION = 16
        LAM_BODY = 0.01
        LAM_ALIGNMENT = 0.02
        logger.log(f"{ITEM=}", f"{RADIUS=}", f"{RESOLUTION=}")

        frame_name = localize_from_images(config, vis_block=False)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        loc_timer_start = time.time_ns()
        logger.log("Starting 3D Instance Segmentation and Object Localization")
        item_cloud, environment_cloud = get_item_pcd(
            ITEM, config, vis_block=True, min_mask_confidence=0.0
        )

        lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, RADIUS)
        logger.log("Ending 3D Instance Segmentation and Object Localization")
        loc_timer_end = time.time_ns()

        item_center = Pose3D(np.mean(np.asarray(item_cloud.points), axis=0))
        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        grasp_timer_start = time.time_ns()
        robot.logger.info("Starting graspnet request.")
        tf_matrices, widths, scores = graspnet_interface.predict_full_grasp(
            item_cloud,
            lim_env_cloud,
            config,
            robot.logger,
            rotation_resolution=RESOLUTION,
            top_n=2,
            n_best=50,
            vis_block=True,
        )
        robot.logger.info("Ending graspnet request.")
        grasp_timer_end = time.time_ns()
        item_center = Pose3D.from_matrix(tf_matrices[0])

        body_planner_start = time.time_ns()
        logger.log("Starting body planning.")
        body_scores = body_planning(
            environment_cloud,
            item_center,
            resolution=16,
            nr_circles=2,
            min_distance=0.65,
            max_distance=0.8,
            floor_height_thresh=0.1,
            n_best=20,
            body_height=0.3,
            vis_block=True,
        )
        logger.log("Ending body planning.")
        body_planner_end = time.time_ns()

        joint_start = time.time_ns()
        logger.log("Starting joint optimization.")
        best_grasp, best_pose, width = joint_optimization_vec(
            item_center,
            tf_matrices,
            widths,
            scores,
            body_scores,
            lambda_body=LAM_BODY,
            lambda_alignment=LAM_ALIGNMENT,
        )
        logger.log("Ending joint optimization.")
        joint_end = time.time_ns()

        print(f"{best_grasp=}", f"{best_pose=}")
        # correct tf_matrix, we need to rotate by 90 degrees
        correct_roll_matrix = Rotation.from_euler(
            "xyz", (-90, 0, 0), degrees=True
        ).as_matrix()
        roll = Pose3D(rot_matrix=correct_roll_matrix)
        grasp_pose = best_grasp @ roll

        direction = grasp_pose.coordinates - best_pose.coordinates
        best_pose.set_rot_from_direction(direction)

        distance_start = 0.25
        robot_to_target = best_grasp.coordinates - best_pose.coordinates
        robot_to_target = robot_to_target / np.linalg.norm(robot_to_target)
        direction = best_grasp.direction()
        extra_offset = np.dot(robot_to_target, direction) * distance_start
        print(f"{extra_offset=}")

        body_pose_adder = Pose3D((-extra_offset, 0, 0))
        body_pose_distanced = best_pose @ body_pose_adder
        body_move_start = time.time_ns()
        logger.log("Starting body movement.")
        print(f"{body_pose_distanced=}")
        move_body(body_pose_distanced.to_dimension(2), frame_name)
        logger.log("Ending body movement.")
        body_move_end = time.time_ns()

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        grasp_pose_new = align_grasp_with_body(best_pose, grasp_pose)
        pose_adder = Pose3D((0, 0, -0.01))
        grasp_pose_new = grasp_pose_new @ pose_adder
        print(f"{grasp_pose_new.coordinates=}")
        print(f"{grasp_pose_new.rot_matrix=}")

        time.sleep(10)

        arm_move_start = time.time_ns()
        logger.log("Starting arm movement.")
        carry_arm(True)
        positional_grab(
            grasp_pose_new,
            distance_start,
            -0.03,
            frame_name,
            already_gripping=False,
        )
        logger.log("Ending arm movement.")
        arm_move_end = time.time_ns()
        # move_arm_distanced(grasp_pose_new, 0.03, frame_name)
        # time.sleep(1)

        time.sleep(5)
        set_gripper(True)
        time.sleep(2)

        stow_arm()
        for name, start, end in (
            ("loc", loc_timer_start, loc_timer_end),
            ("grasp", grasp_timer_start, grasp_timer_end),
            ("body_plan", body_planner_start, body_planner_end),
            ("joint", joint_start, joint_end),
            ("body_move", body_move_start, body_move_end),
            ("arm_move", arm_move_start, arm_move_end),
        ):
            logger.log(f"{name} timer = {end - start} ns")
        logger.log("Ending script.")
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_BetterGrasp(), body_assist=True)


if __name__ == "__main__":
    main()
