# pylint: disable-all
from __future__ import annotations

import numpy as np

from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client import Sdk
from robot_utils.advanced_movement import pull, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    localize_from_images,
    select_points_from_bounding_box,
)
from utils.camera_geometry import plane_fitting
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
    matches: list[Match], depth_response: ImageResponse, frame_name: str
) -> list[Pose3D]:
    """
    Calculates pose and axis of motion of all handles in the image.
    """
    centers = []
    drawer_boxes = [match.drawer.bbox for match in matches]
    handle_boxes = [match.handle.bbox for match in matches]

    # determine center coordinates for all handles
    for handle_bbox in handle_boxes:
        x_center = int((handle_bbox.xmin + handle_bbox.xmax) // 2)
        y_center = int((handle_bbox.ymin + handle_bbox.ymax) // 2)
        center = np.array([x_center, y_center])
        centers.append(center)
    centers = np.stack(centers, axis=0)

    # use centers to get depth and position of handle in frame coordinates
    center_coordss = frame_coordinate_from_depth_image(
        depth_image=depth_response[0],
        depth_image_response=depth_response[1],
        pixel_coordinatess=centers,
        frame_name=frame_name,
    ).reshape((-1, 3))

    # select all points within the point cloud that belong to a drawer (not a handle) and determine the planes
    # the axis of motion is simply the normal of that plane
    drawer_bbox_pointss = select_points_from_bounding_box(
        depth_response, drawer_boxes, frame_name, vis_block=False
    )
    handle_bbox_pointss = select_points_from_bounding_box(
        depth_response, handle_boxes, frame_name, vis_block=False
    )
    points_camera = drawer_bbox_pointss[0]
    drawer_masks = drawer_bbox_pointss[1]
    handle_masks = handle_bbox_pointss[1]
    drawer_only_masks = drawer_masks & (~handle_masks)

    # we use the current body position to get the normal that points towards the robot, not away
    current_body_bosdyn = frame_transformer.get_current_body_position_in_frame(
        frame_name
    )
    current_body = Pose2D.from_bosdyn_pose(current_body_bosdyn)
    poses = []
    for center_coords, bbox_mask in zip(center_coordss, drawer_masks):
        normal = plane_fitting(points_camera[bbox_mask], threshold=0.04)
        # dot product between offset and normal is negative when they point in opposite directions
        offset = center_coords[:2] - current_body.coordinates
        sign = np.sign(np.dot(offset, normal[:2]))
        normal = sign * normal
        pose = Pose3D(center_coords)
        pose.set_rot_from_direction(normal)
        print(pose)
        poses.append(pose)

    return poses


class _DynamicDrawers(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        STAND_DISTANCE = 1.1
        START_BODY = (2.0, -0.5)
        START_ANGLE = 180 + 0
        CABINET_COORDINATES = (0.23, -1.58, 0.4)
        STIFFNESS_DIAG1 = [200, 500, 500, 60, 50, 60]
        STIFFNESS_DIAG2 = [100, 0, 0, 60, 30, 30]
        DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        FORCES = [0, 0, 0, 0, 0, 0]

        frame_name = localize_from_images(config)
        start_pose = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )
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
        stow_arm()
        predictions = drawer_predict(
            color_response[0], config, input_format="bgr", vis_block=True
        )
        matches = drawer_handle_matches(predictions)
        filtered_matches = [
            m for m in matches if (m.handle is not None and m.drawer is not None)
        ]
        filtered_sorted_matches = sorted(
            filtered_matches, key=lambda m: (m.handle.bbox.ymin, m.handle.bbox.xmin)
        )
        handle_poses = calculate_handle_poses(
            filtered_sorted_matches, depth_response, frame_name
        )

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        for handle_pose in handle_poses:
            body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            pull_start, pull_end = pull(
                pose=handle_pose,
                start_distance=0.1,
                mid_distance=-0.1,
                end_distance=0.3,
                frame_name=frame_name,
                stiffness_diag_in=STIFFNESS_DIAG1,
                damping_diag_in=DAMPING_DIAG,
                stiffness_diag_out=STIFFNESS_DIAG2,
                damping_diag_out=DAMPING_DIAG,
                forces=FORCES,
                release_after=True,
            )
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
            )

        stow_arm()

        ###############################################################################
        ################################## FIND ITEM ##################################
        ###############################################################################

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        return frame_name


# TODO: check if I can find pose (so that for pushing I dont go from below or sth)
# TODO: refinement when close to handle
# TODO: take multiple images of cabinet
# TODO: take only plane around handle for plane estimation


def main():
    config = Config()
    take_control_with_function(config, function=_DynamicDrawers(), return_to_start=True)


if __name__ == "__main__":
    main()
