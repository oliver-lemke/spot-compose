from __future__ import annotations

import time

from bosdyn.client import Sdk
from robot_utils.advanced_movement import move_body_distanced, pushing
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from scipy.spatial.transform import Rotation
from utils.coordinates import Pose3D
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


class _Push_Floor(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        BUTTON_COORDINATES = (0.67, -1.52, -0.1)
        RPY = (0, 90, 120)
        STAND_DISTANCE = 0.75
        PUSH_DISTANCES = 0.1, -0.03
        SLEEP_FOR_SAFETY = True

        frame_name = localize_from_images(config)

        ###############################################################################
        ################################## MOVEMENT ###################################
        ###############################################################################

        button_pose = Pose3D(BUTTON_COORDINATES)
        button_pose.set_rot_from_rpy(RPY, degrees=True)

        pitch_matrix = Rotation.from_euler(
            "xyz", (0, 90, 180), degrees=True
        ).as_matrix()
        pitch = Pose3D(rot_matrix=pitch_matrix)
        button_pose_upright = button_pose.copy() @ pitch

        move_body_distanced(
            button_pose_upright.to_dimension(2),
            distance=STAND_DISTANCE,
            frame_name=frame_name,
        )

        if SLEEP_FOR_SAFETY:
            time.sleep(3)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        carry_arm(True)
        pushing(button_pose, *PUSH_DISTANCES, frame_name=frame_name)
        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Floor(), body_assist=True)


if __name__ == "__main__":
    main()
