from __future__ import annotations

import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.advanced_movement import positional_grab
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_body, set_gripper, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from utils.coordinates import Pose2D, Pose3D
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
        ITEM = "green watering can"
        # ITEM = "poro plushy"

        if ITEM == "green watering can":
            grasp_coords = (0.87, -2.33, 0.19024221)
            grasp_rot = np.asarray(
                [
                    [0.05824192, -0.654947, -0.75342713],
                    [0.13504236, -0.7426025, 0.65597647],
                    [-0.98912662, -0.13994989, 0.04519501],
                ]
            )
            body_pose_distanced = Pose3D((1.38, -1.79, 0.30))
        elif ITEM == "poro plushy":
            grasp_coords = (2.4, -2.5, 0.47)
            grasp_rot = np.asarray(
                [
                    [0.03121967, 0.60940186, -0.79224664],
                    [-0.99666667, -0.0407908, -0.07065172],
                    [-0.07537166, 0.79181155, 0.60609703],
                ]
            )
            body_pose_distanced = Pose3D((2.31, -1.33, 0.27))
        else:
            raise ValueError("Wrong ITEM!")

        RADIUS = 0.75
        RESOLUTION = 16
        logger.log(f"{ITEM=}", f"{RADIUS=}", f"{RESOLUTION=}")

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

        grasp_pose_new = Pose3D(grasp_coords, grasp_rot)

        target_to_robot = np.asarray(grasp_coords) - body_pose_distanced.coordinates
        body_pose_distanced.set_rot_from_direction(target_to_robot)

        move_body(body_pose_distanced.to_dimension(2), frame_name)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        # path = config.get_subpath("tmp")
        # pcd_path = os.path.join(path, "pcd.ply")
        # gripper_path = os.path.join(path, "gripper.ply")
        # pcd = o3d.io.read_point_cloud(pcd_path)
        # mesh = o3d.io.read_triangle_mesh(gripper_path)
        # o3d.visualization.draw_geometries([pcd, mesh])

        carry_arm(True)
        positional_grab(
            grasp_pose_new,
            0.25,
            -0.02,
            frame_name,
            already_gripping=False,
        )
        carry_arm(False)

        body_after = Pose3D((2, -1, 0))
        body_after2 = body_after.copy()
        body_after.rot_matrix = body_pose_distanced.rot_matrix.copy()
        move_body(body_after.to_dimension(2), frame_name)
        move_body(body_after2.to_dimension(2), frame_name)

        time.sleep(5)
        set_gripper(True)
        time.sleep(2)

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_BetterGrasp(), body_assist=True)


if __name__ == "__main__":
    main()
