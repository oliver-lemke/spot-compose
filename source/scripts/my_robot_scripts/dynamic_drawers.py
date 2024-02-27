# pylint: disable-all
from __future__ import annotations

import numpy as np

from bosdyn.client import Sdk
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import (
    carry,
    carry_arm,
    gaze,
    move_arm_distanced,
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
from utils import vis
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.drawer_detection import Match, drawer_handle_matches
from utils.drawer_detection import predict as drawer_predict
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


def calculate_handle_poses(
    matches: list[Match], depth_response, frame_name
) -> list[Pose3D]:
    centers = []
    for match in matches:
        handle_bbox = match.handle.bbox
        x_center = int((handle_bbox.xmin + handle_bbox.xmax) // 2)
        y_center = int((handle_bbox.ymin + handle_bbox.ymax) // 2)
        center = np.array([x_center, y_center])
        centers.append(center)
        depth_image = depth_response[0].copy()
        depth_image[:, x_center] = 0
        depth_image[y_center, :] = 0
        vis.show_depth_image(depth_image, "Depth Image")
    centers = np.stack(centers, axis=0)
    print(centers)

    center_coordss = frame_coordinate_from_depth_image(
        depth_image=depth_response[0],
        depth_image_response=depth_response[1],
        pixel_coordinatess=centers,
        frame_name=frame_name,
    ).reshape((-1, 3))

    rot_matrix = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    return [Pose3D(center_coords, rot_matrix) for center_coords in center_coordss]


class _DynamicDrawers(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        STAND_DISTANCE = 1.1
        START_BODY = (1.65, -1.75)
        # START_BODY = (2, 0)
        START_ANGLE = 180 + 0
        CABINET_COORDINATES = (0.23, -1.58, 0.4)
        STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 60]
        STIFFNESS_DIAG2 = [100, 0, 0, 60, 60, 60]
        DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        FORCES = [0, 0, 0, 0, 0, 0]

        frame_name = localize_from_images(config)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        ###############################################################################
        ############################### DETECT DRAWERS ################################
        ###############################################################################

        rotation_pose = Pose2D(START_BODY)
        rotation_pose.set_rot_from_angle(START_ANGLE, degrees=True)
        move_body(rotation_pose, frame_name)

        cabinet_pose = Pose3D(CABINET_COORDINATES)
        carry()
        gaze(cabinet_pose, frame_name, gripper_open=True)
        depth_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
        )
        # stow_arm()
        predictions = drawer_predict(
            color_response[0], config, input_format="bgr", vis_block=True
        )
        for prediction in predictions:
            print(prediction)
        matches = drawer_handle_matches(predictions)
        filtered_matches = [
            m for m in matches if (m.handle is not None and m.drawer is not None)
        ]
        handle_poses = calculate_handle_poses(
            filtered_matches, depth_response, frame_name
        )

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

        handle_poses = handle_poses[:1]
        for handle_pose in handle_poses:
            body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            carry_arm(True)
            move_arm_distanced(handle_pose, 0.1, frame_name)
            set_gripper(True)
            move_arm_distanced(handle_pose, -0.2, frame_name, **keywords1)
            set_gripper(False)
            move_arm_distanced(handle_pose, 0.3, frame_name, **keywords2)
            set_gripper(True)

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
