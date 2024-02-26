# pylint: disable-all
from __future__ import annotations

from bosdyn.client import Sdk
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import get_camera_rgbd, localize_from_images
from utils.coordinates import Pose2D, Pose3D
from utils.darknet_interface import drawer_handle_tuples
from utils.darknet_interface import predict as drawer_predict
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


class _DynamicDrawers(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        DISTANCE = 0.9
        START_ANGLE = 180 + 0
        CABINET_COORDINATES = (0.23, -1.48, 0.3)

        frame_name = localize_from_images(config)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        ###############################################################################
        ############################### DETECT DRAWERS ################################
        ###############################################################################

        rotation_pose = Pose2D((2, -1))
        rotation_pose.set_rot_from_angle(START_ANGLE, degrees=True)
        move_body(rotation_pose, frame_name)

        cabinet_pose = Pose3D(CABINET_COORDINATES)
        carry()
        gaze(cabinet_pose, frame_name, gripper_open=True)
        depth_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
        )
        predictions = drawer_predict(
            color_response[0], config, input_format="bgr", vis_block=True
        )
        for prediction in predictions:
            print(prediction)
        drawer_handle_tuples(predictions)

        stow_arm()

        ###############################################################################
        ################################## FIND ITEM ##################################
        ###############################################################################

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_DynamicDrawers(), return_to_start=True)


if __name__ == "__main__":
    main()
