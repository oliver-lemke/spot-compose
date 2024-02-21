from __future__ import annotations

import numpy as np

from bosdyn.client import Robot, Sdk
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from robot_utils import frame_transformer as ft
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import move_body
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from utils.coordinates import Pose2D
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


class _MovementFunction(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        ODOM_FIRST = True

        ft.FrameTransformer()
        if ODOM_FIRST:
            x, y, angle_degrees = 2.9, -1.5, 0
            pose = Pose2D(np.array([x, y]))
            pose.set_rot_from_angle(angle_degrees, degrees=True)
            move_body(
                pose=pose,
                frame_name=VISION_FRAME_NAME,
            )
        frame_name = localize_from_images(config)
        return frame_name


def main():
    config = Config()
    take_control_with_function(
        config, function=_MovementFunction(), return_to_start=True
    )


if __name__ == "__main__":
    main()
