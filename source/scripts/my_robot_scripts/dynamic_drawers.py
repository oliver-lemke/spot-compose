# pylint: disable-all
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client import Sdk
from robot_utils.advanced_movement import pull, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_arm, move_body, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    localize_from_images,
    project_3D_to_2D,
    select_points_from_bounding_box,
)
from sklearn.cluster import DBSCAN
from utils import recursive_config
from utils.camera_geometry import plane_fitting_open3d
from utils.coordinates import Pose2D, Pose3D, average_pose3Ds, pose_distanced
from utils.drawer_detection import drawer_handle_matches
from utils.object_detetion import BBox, Detection, Match
from utils.drawer_detection import predict_yolodrawer as drawer_predict
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

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


def find_plane_normal_pose(
    points: np.ndarray,
    center_coords: np.ndarray,
    current_body: Pose2D,
    threshold: float = 0.04,
    min_samples: int = 3,
    vis_block: bool = False,
) -> Pose3D:
    # TODO: filter for points nearest to center?
    normal = plane_fitting_open3d(
        points, threshold=threshold, min_samples=min_samples, vis_block=vis_block
    )
    # dot product between offset and normal is negative when they point in opposite directions
    offset = center_coords[:2] - current_body.coordinates
    sign = np.sign(np.dot(offset, normal[:2]))
    normal = sign * normal
    pose = Pose3D(center_coords)
    pose.set_rot_from_direction(normal)
    return pose


def calculate_handle_poses(
    matches: list[Match],
    depth_image_response: (np.ndarray, ImageResponse),
    frame_name: str,
) -> list[Pose3D]:
    """
    Calculates pose and axis of motion of all handles in the image.
    """
    centers = []
    drawer_boxes = [match.drawer.bbox for match in matches]
    handle_boxes = [match.handle.bbox for match in matches]

    depth_image, depth_response = depth_image_response

    # determine center coordinates for all handles
    for handle_bbox in handle_boxes:
        xmin, ymin, xmax, ymax = [int(v) for v in handle_bbox]
        # image_patch = depth_image[xmin:xmax, ymin:ymax].squeeze()
        # image_patch[image_patch == 0.0] = 100_000
        # center_flat = np.argmin(depth_image[xmin:xmax, ymin:ymax])
        # center = np.array(np.unravel_index(center_flat, image_patch.shape))
        # center = center.reshape((2,)) + np.array([xmin, ymin]).reshape((2,))

        x_center, y_center = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        center = np.array([x_center, y_center])
        centers.append(center)
    if len(centers) == 0:
        return []
    centers = np.stack(centers, axis=0)

    # use centers to get depth and position of handle in frame coordinates
    center_coordss = frame_coordinate_from_depth_image(
        depth_image=depth_image,
        depth_response=depth_response,
        pixel_coordinatess=centers,
        frame_name=frame_name,
    ).reshape((-1, 3))

    # select all points within the point cloud that belong to a drawer (not a handle) and determine the planes
    # the axis of motion is simply the normal of that plane
    drawer_bbox_pointss = select_points_from_bounding_box(
        depth_image_response, drawer_boxes, frame_name, vis_block=False
    )
    handle_bbox_pointss = select_points_from_bounding_box(
        depth_image_response, handle_boxes, frame_name, vis_block=False
    )
    points_frame = drawer_bbox_pointss[0]
    drawer_masks = drawer_bbox_pointss[1]
    handle_masks = handle_bbox_pointss[1]
    drawer_only_masks = drawer_masks & (~handle_masks)
    # for mask in drawer_only_masks:
    #     vis.show_point_cloud_in_out(points_frame, mask)

    # we use the current body position to get the normal that points towards the robot, not away
    current_body = frame_transformer.get_current_body_position_in_frame(
        frame_name, in_common_pose=True
    )
    poses = []
    for center_coords, bbox_mask in zip(center_coordss, drawer_only_masks):
        pose = find_plane_normal_pose(
            points_frame[bbox_mask],
            center_coords,
            current_body,
            threshold=0.03,
            min_samples=10,
            vis_block=False,
        )
        print(pose)
        poses.append(pose)

    return poses


def cluster_handle_poses(
    handles_posess: list[list[Pose3D]], eps: float = 0.1, min_samples: int = 2
) -> list[Pose3D]:
    handles_poses_flat = [
        handle_pose for handles_poses in handles_posess for handle_pose in handles_poses
    ]
    handle_coords = [handle_pose.coordinates for handle_pose in handles_poses_flat]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(handle_coords)

    cluster_dict = defaultdict(list)
    for idx, label in enumerate(dbscan.labels_):
        handle_pose = handles_poses_flat[idx]
        cluster_dict[str(label)].append(handle_pose)

    avg_poses = []
    for key, cluster in cluster_dict.items():
        if key == -1:
            continue
        avg_pose = average_pose3Ds(cluster)
        avg_poses.append(avg_pose)
    return avg_poses


def refine_handle_position(
    handle_detections: list[Detection],
    prev_pose: Pose3D,
    depth_image_response: (np.ndarray, ImageResponse),
    frame_name: str,
    discard_threshold: int = 100,
) -> Pose3D:
    prev_center = prev_pose.coordinates.reshape((1, 3))
    prev_center_2D = project_3D_to_2D(depth_image_response, prev_center, frame_name)

    if len(handle_detections) == 0:
        warnings.warn("No handles detected in refinement!")
        return prev_pose
    elif len(handle_detections) > 1:
        centers = []
        for det in handle_detections:
            handle_bbox = det.bbox
            x_center = int((handle_bbox.xmin + handle_bbox.xmax) // 2)
            y_center = int((handle_bbox.ymin + handle_bbox.ymax) // 2)
            center = np.array([x_center, y_center])
            centers.append(center)
        centers = np.stack(centers, axis=0)
        closest_new_idx = np.argmin(
            np.linalg.norm(centers - prev_center_2D, axis=1), axis=0
        )
        handle_bbox = handle_detections[closest_new_idx].bbox
        detection_coordinates = centers[closest_new_idx].reshape((1, 2))
    else:
        handle_bbox = handle_detections[0].bbox
        x_center = int((handle_bbox.xmin + handle_bbox.xmax) // 2)
        y_center = int((handle_bbox.ymin + handle_bbox.ymax) // 2)
        detection_coordinates = np.array([x_center, y_center]).reshape((1, 2))

        # if the distance between expected and mean detection is too large, it likely means that we detect another
        # and also do not detect the original one we want to detect -> discard
        pixel_offset = np.linalg.norm(detection_coordinates - prev_center_2D)
        if pixel_offset > discard_threshold:
            print(
                "Only detection discarded as unlikely to be true detection!",
                f"{pixel_offset=}",
                sep="\n",
            )
            detection_coordinates = prev_center_2D

    center_coords = frame_coordinate_from_depth_image(
        depth_image=depth_image_response[0],
        depth_response=depth_image_response[1],
        pixel_coordinatess=detection_coordinates,
        frame_name=frame_name,
    ).reshape((3,))

    xmin, ymin, xmax, ymax = handle_bbox
    d = 40
    surrounding_bbox = BBox(xmin - d, ymin - d, xmax + d, ymax + d)
    points_frame, [handle_mask, surr_mask] = select_points_from_bounding_box(
        depth_image_response,
        [handle_bbox, surrounding_bbox],
        frame_name,
        vis_block=False,
    )
    surr_only_mask = surr_mask & (~handle_mask)
    current_body = frame_transformer.get_current_body_position_in_frame(
        frame_name, in_common_pose=True
    )

    # vis_block=False
    # vis.show_point_cloud_in_out(points_frame, surr_only_mask)

    pose = find_plane_normal_pose(
        points_frame[surr_only_mask],
        center_coords,
        current_body,
        threshold=0.01,
        min_samples=10,
        vis_block=False,
    )
    print(f"refined pose={pose}")
    return pose


class _DynamicDrawers(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        STAND_DISTANCE = 1.1
        STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 60]
        STIFFNESS_DIAG2 = [100, 0, 0, 60, 30, 30]
        DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
        FORCES = [0, 0, 0, 0, 0, 0]

        config = recursive_config.Config()
        cabinet_pcd, env_pcd = get_mask_points("cabinet", config, idx=2, vis_block=True)
        cabinet_center = np.mean(np.asarray(cabinet_pcd.points), axis=0)
        cabinet_pose = Pose3D(cabinet_center)
        print(f"{cabinet_pose=}")

        body_poses = body_planning_mult_furthest(
            env_pcd,
            cabinet_pose,
            min_target_distance=1.75,
            max_target_distance=2.25,
            min_obstacle_distance=0.75,
            n=4,
            vis_block=True,
        )

        frame_name = localize_from_images(config)
        start_pose = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )
        print(f"{start_pose=}")

        ###############################################################################
        ############################### DETECT DRAWERS ################################
        ###############################################################################

        handles_posess = []
        for start_body in body_poses:
            rotation_pose = start_body.to_dimension(2)
            move_body(rotation_pose, frame_name)

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
            handles_posess.append(handle_poses)

        handle_poses = cluster_handle_poses(handles_posess)
        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################
        camera_add_pose = Pose3D((-0.35, 0, 0.2))
        camera_add_pose.set_rot_from_rpy((0, 35, 0), degrees=True)

        carry()
        for handle_pose in handle_poses:
            body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            # refine
            move_arm(handle_pose @ camera_add_pose, frame_name)
            depth_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
            )
            predictions = drawer_predict(
                color_response[0], config, input_format="bgr", vis_block=True
            )
            handle_detections = [det for det in predictions if det.name == "handle"]
            refined_pose = refine_handle_position(
                handle_detections, handle_pose, depth_response, frame_name
            )

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
