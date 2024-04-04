from __future__ import annotations

import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import move_arm, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.logger import LoggerSingleton
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


class _BetterGrasp(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        frame_name = localize_from_images(config, vis_block=False)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        time.sleep(3)

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################
        grasp_coords = (0.58, -1.36, -0.11)
        grasp_pose = Pose3D(grasp_coords)
        grasp_pose.set_rot_from_rpy((150, 90, 0), degrees=True)
        grasp_pose_distanced = pose_distanced(grasp_pose, 0.15)

        body_pose_distanced = Pose3D((1.12, -0.74, 0))
        target_to_robot = np.asarray(grasp_coords) - body_pose_distanced.coordinates
        body_pose_distanced.set_rot_from_direction(target_to_robot)

        move_body(body_pose_distanced.to_dimension(2), frame_name)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        move_arm(grasp_pose_distanced, frame_name, body_assist=True, timeout=2)
        move_arm(grasp_pose, frame_name, body_assist=True, timeout=4)

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_BetterGrasp(), body_assist=True)


if __name__ == "__main__":
    main()
