# pylint: disable-all
from __future__ import annotations

import os
import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.advanced_movement import (
    dynamically_refined_grasp_renew_grasp,
    move_body_distanced,
    positional_grab,
)
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import (
    carry_arm,
    move_arm,
    move_body,
    set_gripper,
    stow_arm,
)
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    localize_from_images,
)
from scipy.spatial.transform import Rotation
from utils import graspnet_interface
from utils.coordinates import Pose2D, Pose3D
from utils.mask3D_interface import get_coordinates_from_item
from utils.point_clouds import body_planning, get_radius_env_cloud
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.user_input import get_wanted_item_mask3d
from utils.vitpose_interface import predict

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


class _HumanStream(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        NR_ROTATIONS = 8
        CONFIDENCE_THRESHOLD = 0.5
        DISTANCE = 0.9
        START_ANGLE = 180 + 0
        ITEM, INDEX, BOOL = "bag", 0, False
        ITEM, INDEX, BOOL = "lamp", 2, False
        # TARGET_BODY_PART = BodyPart.RIGHT_HAND

        frame_name = localize_from_images(config)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        ###############################################################################
        ################################ DETECT HUMAN #################################
        ###############################################################################

        rotation_pose = Pose2D((2, -1))
        increments = 360 / NR_ROTATIONS
        human_pose = None
        for i in range(NR_ROTATIONS):
            angle = START_ANGLE + i * increments
            rotation_pose.set_rot_from_angle(angle, degrees=True)
            move_body(rotation_pose, frame_name)
            depth_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
            )

            humans = predict(
                color_response[0], config, input_format="bgr", vis_block=False
            )
            if humans is None or len(humans) == 0:
                # no humans detected
                continue
            detection = humans[0, 2, 1:].astype(int)

            center_coords = frame_coordinate_from_depth_image(
                depth_image=depth_response[0],
                depth_image_response=depth_response[1],
                pixel_coordinates=detection.reshape((1, 2)),
                frame_name=frame_name,
            ).reshape((3,))
            human_pose = Pose3D(center_coords)

            set_gripper(False)
            break

        if human_pose is None:
            raise RuntimeError("No human detected!")

        ###############################################################################
        ################################## FIND ITEM ##################################
        ###############################################################################

        # get position of wanted item
        if ITEM is None:
            item = get_wanted_item_mask3d()
        else:
            item = str(ITEM)

        mask_path = config.get_subpath("masks")
        ending = config["pre_scanned_graphs"]["masked"]
        mask_path = os.path.join(mask_path, ending)

        pc_path = config.get_subpath("aligned_point_clouds")
        ending = config["pre_scanned_graphs"]["high_res"]
        pc_path = os.path.join(str(pc_path), ending, "scene.ply")

        item_cloud, environment_cloud = get_coordinates_from_item(
            item, mask_path, pc_path, INDEX
        )
        lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, 0.5)
        end_coordinates_np = np.mean(np.asarray(item_cloud.points), axis=0)
        item_coordinates = Pose3D(end_coordinates_np)
        print(f"{item_coordinates=}")

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        robot.logger.info("Starting body planning.")
        robot_target = body_planning(
            environment_cloud, item_coordinates, min_distance=0.65, max_distance=0.65
        )[0]

        robot.logger.info("Starting graspnet request.")
        tf_matrix, _ = graspnet_interface.predict_full_grasp(
            item_cloud,
            lim_env_cloud,
            config,
            robot.logger,
            rotation_resolution=16,
            top_n=2,
            vis_block=False,
        )

        direction = item_coordinates.coordinates - robot_target.coordinates
        robot_target.set_rot_from_direction(direction)

        # start_pose_bd = transformer.get_current_body_position_in_frame(frame_name)
        # start_pose = Pose2D.from_bosdyn_pose(start_pose_bd)
        # confirm_move(start_pose, robot_target.to_dimension(2))
        move_body(
            robot_target.to_dimension(2),
            frame_name,
        )

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        grasp_pose = Pose3D.from_matrix(tf_matrix)
        carry_arm(True)
        # unstow_arm(robot, robot_command_client, True)

        # correct tf_matrix, we need to rotate by 90 degrees
        correct_roll_matrix = Rotation.from_euler(
            "xyz", (-90, 0, 0), degrees=True
        ).as_matrix()
        roll = Pose3D(rot_matrix=correct_roll_matrix)
        grasp_pose = grasp_pose @ roll

        # final_roll_value, *_ = Rotation.from_matrix(
        #     grasp_pose.as_matrix()[:3, :3]
        # ).as_euler("xyz", degrees=True)
        # final_roll_value = final_roll_value % 360
        if BOOL:
            roll_matrix = Rotation.from_euler("xyz", (180, 0, 0), degrees=True)
            grasp_pose = grasp_pose @ Pose3D.from_scipy_rotation(roll_matrix)

        positional_grab(
            grasp_pose,
            0.1,
            -0.05,
            frame_name,
            already_gripping=False,
        )

        carry_arm()

        ###############################################################################
        ################################ MOVE TO HUMAN ################################
        ###############################################################################

        robot_target = body_planning(
            environment_cloud, human_pose, min_distance=DISTANCE, max_distance=DISTANCE
        )[0]
        direction = human_pose.coordinates - robot_target.coordinates
        robot_target.set_rot_from_direction(direction)
        move_body(robot_target.to_dimension(2), frame_name)

        # human_pose.rot_matrix = tf_matrix[:3, :3]
        # move_arm(human_pose, frame_name, gripper_open=False)
        time.sleep(1)
        set_gripper(True)
        time.sleep(3)
        set_gripper(False)
        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_HumanStream(), return_to_start=True)


if __name__ == "__main__":
    main()
