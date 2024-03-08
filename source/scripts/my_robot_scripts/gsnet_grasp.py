from __future__ import annotations

import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.advanced_movement import positional_grab, adapt_grasp
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_body, set_gripper, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from scipy.spatial.transform import Rotation
from utils import graspnet_interface
from utils.coordinates import Pose2D, Pose3D
from utils.openmask_interface import get_mask_points
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
from utils.logger import LoggerSingleton

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()
logger = LoggerSingleton()


class _BetterGrasp(ControlFunction):
    def __call__(
            self,
            config: Config,
            sdk: Sdk,
            *args,
            **kwargs,
    ) -> str:
        ITEM = "dark green bottle"
        RADIUS = 0.75
        RESOLUTION = 16
        STIFFNESS_DIAG = [100, 500, 500, 60, 60, 60]
        DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        FORCES = [0, 0, 0, 0, 0, 0]
        logger.log(f"{ITEM=}", f"{RADIUS=}", f"{RESOLUTION=}")

        frame_name = localize_from_images(config)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        logger.log(f"{start_pose=}")

        logger.log("Starting 3D Instance Segmentation and Object Localization")
        item_cloud, environment_cloud = get_mask_points(ITEM, config, vis_block=False)


        lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, RADIUS)
        logger.log("Ending 3D Instance Segmentation and Object Localization")

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        robot.logger.info("Starting graspnet request.")
        tf_matrix, _ = graspnet_interface.predict_full_grasp(
            item_cloud,
            lim_env_cloud,
            config,
            robot.logger,
            rotation_resolution=RESOLUTION,
            top_n=2,
            vis_block=False,
        )
        robot.logger.info("Ending graspnet request.")
        grasp_pose = Pose3D.from_matrix(tf_matrix)

        # correct tf_matrix, we need to rotate by 90 degrees
        correct_roll_matrix = Rotation.from_euler(
            "xyz", (-90, 0, 0), degrees=True
        ).as_matrix()
        roll = Pose3D(rot_matrix=correct_roll_matrix)
        grasp_pose = grasp_pose @ roll

        logger.log("Starting body planning.")
        robot_target = body_planning(
            environment_cloud, grasp_pose, min_distance=0.8, max_distance=0.8, floor_height_thresh=0.1
        )[0]
        logger.log("Ending body planning.")

        direction = grasp_pose.coordinates - robot_target.coordinates
        robot_target.set_rot_from_direction(direction)

        logger.log("Starting body movement.")
        move_body(robot_target.to_dimension(2), frame_name)
        logger.log("Ending body movement.")

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################


        logger.log("Starting arm movement.")
        carry_arm(True)
        grasp_pose_new = adapt_grasp(robot_target, grasp_pose)
        positional_grab(
            grasp_pose_new,
            0.1,
            -0.1,
            frame_name,
            already_gripping=False,
            stiffness_diag = STIFFNESS_DIAG,
            damping_diag = DAMPING_DIAG,
            forces = FORCES,
        )
        logger.log("Ending arm movement.")

        carry_arm(False)

        time.sleep(5)
        set_gripper(True)
        time.sleep(2)

        stow_arm()
        logger.log("Ending script.")
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_BetterGrasp(), body_assist=True)


if __name__ == "__main__":
    main()
