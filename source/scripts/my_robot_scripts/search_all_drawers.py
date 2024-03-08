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
    GRIPPER_IMAGE_COLOR,
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    get_rgb_pictures,
    localize_from_images,
    project_3D_to_2D,
    select_points_from_bounding_box,
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
ITEMS = ['deer toy', 'small clock', 'headphones', 'watch', 'highlighter', 'red bottle']


def determine_handle_center(depth_image: np.ndarray, bbox: BBox, approach: str = "center") -> np.ndarray:
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    if approach == "min":
        image_patch = depth_image[xmin:xmax, ymin:ymax].squeeze()
        image_patch[image_patch == 0.0] = 100_000
        center_flat = np.argmin(depth_image[xmin:xmax, ymin:ymax])
        center = np.array(np.unravel_index(center_flat, image_patch.shape))
        center = center.reshape((2,)) + np.array([xmin, ymin]).reshape((2,))
    elif approach == "center":
        x_center, y_center = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        center = np.array([x_center, y_center])
    else:
        raise ValueError(f"Unknown type {approach}. Must be either 'min' or 'center'.")
    return center


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
        center = determine_handle_center(depth_image, handle_bbox)
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
        poses.append(pose)

    return poses


def cluster_handle_poses(
        handles_posess: list[list[Pose3D]], eps: float = MIN_PAIRWISE_DRAWER_DISTANCE, min_samples: int = 2
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
    print("cluster_dict=", *cluster_dict.items(), sep="\n")

    avg_poses = []
    for key, cluster in cluster_dict.items():
        if key == "-1":
            continue
        avg_pose = average_pose3Ds(cluster)
        avg_poses.append(avg_pose)
    return avg_poses


def refine_handle_position(
        handle_detections: list[Detection],
        prev_pose: Pose3D,
        depth_image_response: (np.ndarray, ImageResponse),
        frame_name: str,
        discard_threshold: int = MIN_PAIRWISE_DRAWER_DISTANCE,
) -> (Pose3D, bool):
    depth_image, depth_response = depth_image_response
    prev_center_3D = prev_pose.coordinates.reshape((1, 3))

    if len(handle_detections) == 0:
        warnings.warn("No handles detected in refinement!")
        return prev_pose, True
    elif len(handle_detections) > 1:
        centers_2D = []
        for det in handle_detections:
            handle_bbox = det.bbox
            center = determine_handle_center(depth_image, handle_bbox)
            centers_2D.append(center)
        centers_2D = np.stack(centers_2D, axis=0)
        centers_3D = frame_coordinate_from_depth_image(depth_image, depth_response, centers_2D, frame_name)
        closest_new_idx = np.argmin(
            np.linalg.norm(centers_3D - prev_center_3D, axis=1), axis=0
        )
        handle_bbox = handle_detections[closest_new_idx].bbox
        detection_coordinates_3D = centers_3D[closest_new_idx].reshape((1, 3))
    else:
        handle_bbox = handle_detections[0].bbox
        center = determine_handle_center(depth_image, handle_bbox).reshape((1, 2))
        detection_coordinates_3D = frame_coordinate_from_depth_image(depth_image, depth_response,
                                                                     center, frame_name)

    # if the distance between expected and mean detection is too large, it likely means that we detect another
    # and also do not detect the original one we want to detect -> discard
    handle_offset = np.linalg.norm(detection_coordinates_3D - prev_center_3D)
    discarded = False
    if handle_offset > discard_threshold:
        discarded = True
        print(
            "Only detection discarded as unlikely to be true detection!",
            f"{handle_offset=}",
            sep="\n",
        )
        detection_coordinates_3D = prev_center_3D

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

    # vis_block = False
    # vis.show_point_cloud_in_out(points_frame, surr_only_mask)

    detection_coordinates_3D = detection_coordinates_3D.reshape((3,))
    pose = find_plane_normal_pose(
        points_frame[surr_only_mask],
        detection_coordinates_3D,
        current_body,
        threshold=0.04,
        min_samples=10,
        vis_block=False,
    )
    return pose, discarded


def filter_handle_poses(handle_poses: list[Pose3D]):
    return [p for p in handle_poses if 0.05 < p.coordinates[-1] < 0.75]


def search_drawer(
        cabinet_poses: list[Pose3D], env_pcd: PointCloud, config: Config, frame_name: str
) -> list[tuple[Pose3D, Detection]]:
    gaze_and_bodies = []
    for cabinet_pose in cabinet_poses:
        body_poses = body_planning_mult_furthest(
            env_pcd,
            cabinet_pose,
            min_target_distance=1.75,
            max_target_distance=2.25,
            min_obstacle_distance=0.75,
            n=4,
            vis_block=False,
        )
        gaze_and_body = [(cabinet_pose, body_pose) for body_pose in body_poses]
        gaze_and_bodies.extend(gaze_and_body)

    ###############################################################################
    ############################### DETECT DRAWERS ################################
    ###############################################################################
    camera_add_pose = Pose3D(CAMERA_ADD_COORDS)
    camera_add_pose.set_rot_from_rpy((0, CAMERA_ANGLE, 0), degrees=True)

    handle_posess = []
    for cabinet_pose, body_pose in gaze_and_bodies:
        rotation_pose = body_pose.to_dimension(2)
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
        handle_posess.append(handle_poses)

    print("all detections:", *handle_posess, sep="\n")
    handle_poses = cluster_handle_poses(handle_posess, eps=MIN_PAIRWISE_DRAWER_DISTANCE)
    print("clustered:", *handle_poses, sep="\n")
    handle_poses = filter_handle_poses(handle_poses)
    print("filtered:", *handle_poses, sep="\n")

    ###############################################################################
    ################################ ARM COMMANDS #################################
    ###############################################################################
    camera_add_pose_refinement_right = Pose3D((-0.35, -0.2, 0.15))
    camera_add_pose_refinement_right.set_rot_from_rpy((0, 25, 35), degrees=True)
    camera_add_pose_refinement_left = Pose3D((-0.35, 0.2, 0.15))
    camera_add_pose_refinement_left.set_rot_from_rpy((0, 25, -35), degrees=True)
    ref_add_poses = (camera_add_pose_refinement_right, camera_add_pose_refinement_left)

    detection_drawer_pairs = []

    carry()
    for handle_pose in handle_poses:
        # if no handle is detected in the refinement, redo it from a different position
        body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
        move_body(body_pose, frame_name)

        refined_pose = handle_pose
        for ref_pose in ref_add_poses:
            move_arm(handle_pose @ ref_pose, frame_name)
            depth_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
            )
            predictions = drawer_predict(
                color_response[0], config, input_format="bgr", vis_block=True
            )
            handle_detections = [det for det in predictions if det.name == "handle"]
            refined_pose, discarded = refine_handle_position(
                handle_detections, handle_pose, depth_response, frame_name
            )
            if not discarded:
                break

        print(f"{refined_pose=}")

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
        detections = detect_objects(imgs[0][0], ITEMS, vis_block=True)
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

    stow_arm()
    return detection_drawer_pairs


def calculate_projected_area(points: np.ndarray) -> float:
    """Calculate the projected area of points on the XY plane."""
    hull = ConvexHull(points[:, :2])  # Use only X and Y for convex hull
    return hull.volume  # For 2D convex hulls, `volume` is the area


def calculate_center(points: np.ndarray) -> np.ndarray:
    """Calculate the center of a set of points."""
    return np.mean(points, axis=0)


def split_and_calculate_centers(
        points: np.ndarray, threshold: float
) -> list[np.ndarray]:
    """Split the point cloud and calculate centers if above threshold."""
    projected_area = calculate_projected_area(points)
    print(f"{projected_area=}")
    if projected_area <= threshold:
        return [calculate_center(points)]
    else:
        # Calculate the number of parts to split into, rounded up
        num_parts = int(np.ceil(projected_area / threshold))
        kmeans = KMeans(n_clusters=num_parts)
        labels = kmeans.fit_predict(points[:, :2])  # Fit only using X and Y

        centers = []
        for i in range(num_parts):
            part_points = points[labels == i]
            centers.append(calculate_center(part_points))
        return centers


class _DynamicDrawers(ControlFunction):
    def __call__(
            self,
            config: Config,
            sdk: Sdk,
            *args,
            **kwargs,
    ) -> str:
        indices = (2, 7)
        config = recursive_config.Config()

        frame_name = localize_from_images(config)
        start_pose = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )
        print(f"{start_pose=}")

        draw_det_pairs = []
        for idx in indices:
            cabinet_pcd, env_pcd = get_mask_points(
                "cabinet", config, idx=idx, vis_block=True
            )
            cabinet_centers = split_and_calculate_centers(
                np.asarray(cabinet_pcd.points), threshold=SPLIT_THRESH
            )
            cabinet_poses = [Pose3D(center) for center in cabinet_centers]
            print(f"{cabinet_poses=}")
            pairs = search_drawer(cabinet_poses, env_pcd, config, frame_name)
            draw_det_pairs.extend(pairs)

        print(f"{draw_det_pairs=}")
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_DynamicDrawers(), return_to_start=True)


if __name__ == "__main__":
    main()
