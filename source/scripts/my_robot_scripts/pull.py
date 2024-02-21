from __future__ import annotations

import time

from bosdyn.client import Robot, Sdk
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from robot_utils.advanced_movement import pulling, rotate, rotate_and_move_distanced
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import high_carry, unstow_arm
from robot_utils.graph_nav import full_localize
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.recursive_config import Config


class _Pull(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        robot: Robot,
        robot_command_client: RobotCommandClient,
        robot_state_client: RobotStateClient,
        *args,
        **kwargs,
    ) -> None:
        KNOB_COORDINATES = (1.4, -2.32, 0.54)
        RPY = (0, 0, 180 + 90)
        STAND_DISTANCE = 1
        PULL_DISTANCES = 0.1, 0, 0.15
        SLEEP_FOR_SAFETY = True

        frame_name, transformer, _, _ = full_localize(config, robot, robot_state_client)

        ###############################################################################
        ################################## MOVEMENT ###################################
        ###############################################################################

        knob_pose = Pose3D(KNOB_COORDINATES)
        knob_pose.set_rot_from_rpy(RPY, degrees=True)
        body_pose = pose_distanced(knob_pose, STAND_DISTANCE).to_dimension(2)

        static_params = {
            "frame_name": frame_name,
        }
        rotate_and_move_distanced(body_pose, 0, sleep=SLEEP_FOR_SAFETY, **static_params)
        rotate(knob_pose.to_dimension(2), **static_params)

        if SLEEP_FOR_SAFETY:
            time.sleep(3)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        unstow_arm(robot, robot_command_client, True)

        pulling(knob_pose, *PULL_DISTANCES, release_after=True, **static_params)

        high_carry(transformer, robot, robot_command_client)

        if SLEEP_FOR_SAFETY:
            time.sleep(2)


def main():
    config = Config()
    take_control_with_function(config, function=_Pull(), body_assist=True)


if __name__ == "__main__":
    main()
