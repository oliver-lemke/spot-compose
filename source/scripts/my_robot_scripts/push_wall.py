from __future__ import annotations

import time

from bosdyn.client import Sdk
from robot_utils.advanced_movement import push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.user_input import confirm_move

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


class _Push_Wall(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        BUTTON_COORDINATES = (0.29, -0.48, 0.8)
        RPY = (0, 0, 180)
        STAND_DISTANCE = 1.2
        PUSH_DISTANCES = 0.2, -0.01
        SLEEP_FOR_SAFETY = False

        frame_name = localize_from_images(config)

        ###############################################################################
        ################################## MOVEMENT ###################################
        ###############################################################################

        button_pose = Pose3D(BUTTON_COORDINATES)
        button_pose.set_rot_from_rpy(RPY, degrees=True)

        body_pose = pose_distanced(button_pose, STAND_DISTANCE).to_dimension(2)

        static_params = {
            "frame_name": frame_name,
        }
        start_pose_bd = frame_transformer.get_current_body_position_in_frame(frame_name)
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bd)
        confirm_move(start_pose, body_pose)
        move_body(body_pose, frame_name)

        if SLEEP_FOR_SAFETY:
            time.sleep(3)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        carry_arm(True)
        push(button_pose, *PUSH_DISTANCES, **static_params)
        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Wall(), body_assist=True)


if __name__ == "__main__":
    main()
