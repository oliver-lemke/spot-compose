from __future__ import annotations

import time

from bosdyn.client import Sdk
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import (
    carry_arm,
    move_arm_distanced,
    move_body,
    set_gripper,
    stow_arm,
)
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


class _Search(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        KNOB_COORDINATES_TOP = (0.28, -1.68, 0.52)
        RPY = (0, 0, 225)
        STAND_DISTANCE = 1.1
        SLEEP_FOR_SAFETY = True
        STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 60]
        STIFFNESS_DIAG2 = [100, 0, 0, 60, 60, 60]
        DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        FORCES = [0, 0, 0, 0, 0, 0]

        frame_name = localize_from_images(config)

        ###############################################################################
        ################################## MOVEMENT ###################################
        ###############################################################################

        knob_pose_top = Pose3D(KNOB_COORDINATES_TOP)
        knob_pose_top.set_rot_from_rpy(RPY, degrees=True)
        body_pose = pose_distanced(knob_pose_top, STAND_DISTANCE).to_dimension(2)

        ###############################################################################
        ############################### MOVE COMMANDS #################################
        ###############################################################################

        start_pose_bd = frame_transformer.get_current_body_position_in_frame(frame_name)
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bd)
        confirm_move(start_pose, body_pose)
        move_body(body_pose, frame_name)

        if SLEEP_FOR_SAFETY:
            time.sleep(3)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################
        keywords1 = {
            "stiffness_diag": STIFFNESS_DIAG1,
            "damping_diag": DAMPING_DIAG,
            "forces": FORCES,
            "timeout": 6,
        }
        keywords2 = {
            "stiffness_diag": STIFFNESS_DIAG2,
            "damping_diag": DAMPING_DIAG,
            "forces": FORCES,
            "timeout": 6,
        }

        carry_arm(True)
        move_arm_distanced(knob_pose_top, 0.1, frame_name)
        set_gripper(True)
        move_arm_distanced(knob_pose_top, -0.1, frame_name, **keywords1)
        set_gripper(False)
        move_arm_distanced(knob_pose_top, 0.3, frame_name, **keywords2)
        set_gripper(True)

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Search(), body_assist=True)


if __name__ == "__main__":
    main()
