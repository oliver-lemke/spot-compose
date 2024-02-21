from __future__ import annotations

import time

from bosdyn.client import Sdk
from robot_utils.advanced_movement import pulling, pushing
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_arm, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    GRIPPER_IMAGE_COLOR,
    get_rgb_pictures,
    localize_from_images,
)
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
from utils.user_input import confirm_move, get_n_word_answer
from utils.zero_shot_object_detection import detect_objects

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
        KNOB_COORDINATES_TOP = (0.38, -1.58, 0.52)
        KNOB_COORDINATES_BOT = (0.38, -1.58, 0.32)
        RPY = (0, 0, 180)
        STAND_DISTANCE = 1.1
        PULL_DISTANCES = 0.1, 0, 0.2
        PUSH_DISTANCES = (0.05, -PULL_DISTANCES[-1])
        SLEEP_FOR_SAFETY = True
        CAMERA_ADD_COORDS = (-0.1, 0, 0.3)
        CAMERA_ANGLE = 55

        items = get_n_word_answer("What object are you looking for?")

        frame_name = localize_from_images(config)

        ###############################################################################
        ################################## MOVEMENT ###################################
        ###############################################################################

        knob_pose_top = Pose3D(KNOB_COORDINATES_TOP)
        knob_pose_top.set_rot_from_rpy(RPY, degrees=True)
        knob_pose_bot = Pose3D(KNOB_COORDINATES_BOT)
        knob_pose_bot.set_rot_from_rpy(RPY, degrees=True)
        camera_add_pose = Pose3D(CAMERA_ADD_COORDS)
        camera_add_pose.set_rot_from_rpy((0, CAMERA_ANGLE, 0), degrees=True)
        body_pose = pose_distanced(knob_pose_top, STAND_DISTANCE).to_dimension(2)

        static_params = {
            "frame_name": frame_name,
        }

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
        detection_dicts = {}

        carry_arm(True)
        # First drawer
        *_, pulled_pose = pulling(
            knob_pose_top, *PULL_DISTANCES, release_after=True, **static_params
        )
        move_arm(pulled_pose @ camera_add_pose, **static_params)
        imgs = get_rgb_pictures([GRIPPER_IMAGE_COLOR])
        detection_dicts["top"] = detect_objects(imgs[0][0], items)
        pushing(pulled_pose, *PUSH_DISTANCES, **static_params)

        # Second drawer
        *_, pulled_pose = pulling(
            knob_pose_bot, *PULL_DISTANCES, release_after=True, **static_params
        )
        move_arm(pulled_pose @ camera_add_pose, **static_params)
        imgs = get_rgb_pictures([GRIPPER_IMAGE_COLOR])
        detection_dicts["bottom"] = detect_objects(imgs[0][0], items)
        pushing(pulled_pose, *PUSH_DISTANCES, **static_params)

        output_positions = {}
        for pos, detection_dict in detection_dicts.items():
            for item in items:
                # for each item
                if item in detection_dict:
                    # check if it is in the corresponding image
                    score = detection_dict[item]
                    # and put it in the output if it is the best (or only) detection
                    if (
                        item in output_positions
                        and score <= output_positions[item]["score"]
                    ):
                        continue
                    output_positions[item] = {"score": score, "pos": pos}

        for item in items:
            item_str = item.capitalize()
            if item in output_positions:
                print(f"{item_str} is in the {output_positions[item]['pos']} drawer!")
            else:
                print(f"{item_str} could not be found!")

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Search(), body_assist=True)


if __name__ == "__main__":
    main()
