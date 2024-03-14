# pylint: disable-all
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client import Sdk
from scipy.spatial.transform import Rotation

from robot_utils.advanced_movement import pull, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_arm, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    GRIPPER_IMAGE_COLOR,
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    get_rgb_pictures,
    localize_from_images,
    project_3D_to_2D,
    select_points_from_bounding_box, relocalize,
)
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, KMeans
from utils import recursive_config, vis
from utils.camera_geometry import plane_fitting_open3d
from utils.coordinates import Pose2D, Pose3D, average_pose3Ds, pose_distanced
from utils.drawer_detection import drawer_handle_matches
from utils.drawer_detection import predict_yolodrawer as drawer_predict
from utils.importer import PointCloud
from utils.object_detetion import BBox, Detection, Match
from utils.openmask_interface import get_mask_points
from utils.point_clouds import body_planning_mult_furthest
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.zero_shot_object_detection import detect_objects

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

STAND_DISTANCE = 1.0
STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 45]
STIFFNESS_DIAG2 = [100, 0, 0, 60, 30, 30]
DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
FORCES = [0, 0, 0, 0, 0, 0]
CAMERA_ADD_COORDS = (-0.25, 0, 0.3)
CAMERA_ANGLE = 55
SPLIT_THRESH = 1.0
MIN_PAIRWISE_DRAWER_DISTANCE = 0.1
ITEMS = ["deer toy", "small clock", "headphones", "watch", "highlighter", "red bottle"]

class _DynamicDrawers(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        indices = (3, 4)
        config = recursive_config.Config()

        frame_name = localize_from_images(config)
        start_pose = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )
        print(f"{start_pose=}")


        camera_add_pose = Pose3D(CAMERA_ADD_COORDS)
        camera_add_pose.set_rot_from_rpy((0, CAMERA_ANGLE, 0), degrees=True)

        handle_poses = [Pose3D((-0.05, -1.45, 0.52)), Pose3D((-0.05, -1.45, 0.33)), Pose3D((-0.05, -1.45, 0.14))]
        for handle_pose in handle_poses:
            handle_pose.rot_matrix = Rotation.from_euler("z", 180, degrees=True).as_matrix()

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        detection_drawer_pairs = []

        carry()
        for handle_pose in handle_poses:
            # if no handle is detected in the refinement, redo it from a different position
            body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            refined_pose = handle_pose

            print(f"{refined_pose=}")

            relocalize(config, frame_name)
            pull_start, pull_end = pull(
                pose=refined_pose,
                start_distance=0.1,
                mid_distance=-0.1,
                end_distance=0.3,
                frame_name=frame_name,
                stiffness_diag_in=STIFFNESS_DIAG1,
                damping_diag_in=DAMPING_DIAG,
                stiffness_diag_out=STIFFNESS_DIAG2,
                damping_diag_out=DAMPING_DIAG,
                forces=FORCES,
                follow_arm=False,
                release_after=True,
            )
            camera_pose = refined_pose @ camera_add_pose
            print(f"{camera_pose=}")
            move_arm(camera_pose, frame_name=frame_name, body_assist=True)
            imgs = get_rgb_pictures([GRIPPER_IMAGE_COLOR])
            detections = detect_objects(imgs[0][0], ITEMS, vis_block=False)
            print(f"{detections=}")
            pairs = [(refined_pose, det) for det in detections]
            detection_drawer_pairs.extend(pairs)
            direction = pull_start.coordinates - pull_end.coordinates
            pull_start.set_rot_from_direction(direction)
            pull_end.set_rot_from_direction(direction)
            push(
                start_pose=pull_end,
                end_pose=pull_start,
                start_distance=0.1,
                end_distance=-0.05,
                frame_name=frame_name,
                stiffness_diag=STIFFNESS_DIAG1,
                damping_diag=DAMPING_DIAG,
                forces=FORCES,
                follow_arm=False,
            )
            move_arm(pull_end, frame_name)

        stow_arm(False)
        print(f"{detection_drawer_pairs=}")
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_DynamicDrawers(), return_to_start=True)


if __name__ == "__main__":
    main()
