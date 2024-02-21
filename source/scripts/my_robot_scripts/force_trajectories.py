from __future__ import annotations

import time

from bosdyn.client import Sdk
from bosdyn.client.frame_helpers import HAND_FRAME_NAME, VISION_FRAME_NAME
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import move_body, stow_arm, unstow_arm
from robot_utils.frame_transformer import FrameTransformer, FrameTransformerSingleton
from robot_utils.trajectory_movement import move_arm_trajectory
from utils.coordinates import Pose2D, build_trajectory_point
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


class _ForceTrajectory(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        frame_transformer.set_instance(FrameTransformer())
        pose = Pose2D((1, 0))
        pose.set_rot_from_angle(-90, degrees=True)
        move_body(pose, VISION_FRAME_NAME)

        t0 = 0
        t1 = 1
        traj_point0 = build_trajectory_point(t0)
        traj_point1 = build_trajectory_point(t1, torque_x=2)
        traj_points = [traj_point0, traj_point1]

        move_arm_trajectory(
            traj_points, HAND_FRAME_NAME, unstow=True, stow=False, body_assist=True
        )

        time.sleep(5.0 + t1)
        unstow_arm()
        stow_arm()

        return HAND_FRAME_NAME


def main():
    config = Config()
    take_control_with_function(
        config, function=_ForceTrajectory(), body_assist=True, return_to_start=False
    )


if __name__ == "__main__":
    main()
