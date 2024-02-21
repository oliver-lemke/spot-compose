from __future__ import annotations

import json
import os
import re
from collections import OrderedDict

import numpy as np

import apriltag
import cv2
import open3d as o3d
from bosdyn.client import math_helpers
from bosdyn.client.math_helpers import Quat
from scipy.spatial.transform import Rotation
from scripts.point_cloud_scripts.vis_ply_point_clouds_with_coordinates import draw_cloud
from utils import recursive_config
from utils.point_clouds import add_coordinate_system


def fetch_paths(
    directory_path: bytes | str,
) -> (dict[int, dict[str, bytes]], bytes):
    """
    Returns the paths to all relevant files for the scan alignment.
    Namely, all images, the associated json files, and the point cloud
    :param directory_path: The path where all images, json files, and the ply file are
    stored
    :return: a dictionary containing the paths to both images and jsons keyed by number,
     and the path to the ply file
    """
    jpg_pattern = r"^frame_(\d{5})\.jpg$"
    json_pattern = r"^frame_(\d{5})\.json$"
    jpg_json_paths = OrderedDict()

    # Ensure the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Loop through the files in the directory
        files = os.listdir(directory_path)
        files.sort()
        for filename in files:
            file_path = os.path.join(directory_path, filename)

            # Check if the file matches the JPG pattern
            jpg_match = re.match(jpg_pattern, str(filename))
            if jpg_match:
                frame_name = int(jpg_match.group(1))
                if frame_name in jpg_json_paths:
                    jpg_json_paths[frame_name]["jpg"] = file_path
                else:
                    jpg_json_paths[frame_name] = {"jpg": file_path}

            # Check if the file matches the JSON pattern
            json_match = re.match(json_pattern, str(filename))
            if json_match:
                frame_name = int(json_match.group(1))
                if frame_name in jpg_json_paths:
                    jpg_json_paths[frame_name]["json"] = file_path
                else:
                    jpg_json_paths[frame_name] = {"json": file_path}

    ply_path = os.path.join(str(directory_path), "point_cloud.ply")
    assert os.path.exists(ply_path), f"PLY file does not exists at {ply_path}!"

    return jpg_json_paths, ply_path


def get_best_detection(jpg_json_paths, tag_id: int = 51):
    options = apriltag.DetectorOptions(families="tag36h11", refine_pose=True)
    detector = apriltag.Detector(options)

    best_fit = -1
    best_frame_number = -1
    best_detection = None

    for frame_nr, jpg_json_dict in jpg_json_paths.items():
        # get data (image and camera intrinsics)
        jpg_path = jpg_json_dict["jpg"]
        img = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)

        detections = detector.detect(img)
        for detection in detections:
            if detection.tag_id != tag_id:
                continue
            if detection.decision_margin > best_fit:
                best_fit = detection.decision_margin
                best_frame_number = frame_nr
                best_detection = detection
                break

    return best_frame_number, best_detection


def get_e_coordinates(
    proj_mat: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    zero_tform_camera = math_helpers.SE3Pose.from_matrix(proj_mat)

    camera = math_helpers.SE3Pose(x=0, y=0, z=0, rot=Quat.from_matrix(proj_mat[:3, :3]))
    camera_e1 = math_helpers.SE3Pose(x=1, y=0, z=0, rot=Quat())
    camera_e2 = math_helpers.SE3Pose(x=0, y=1, z=0, rot=Quat())
    camera_e3 = math_helpers.SE3Pose(x=0, y=0, z=1, rot=Quat())

    zero = zero_tform_camera * camera
    e1 = zero_tform_camera * camera_e1
    e2 = zero_tform_camera * camera_e2
    e3 = zero_tform_camera * camera_e3

    return (
        np.asarray([zero.x, zero.y, zero.z]),
        np.asarray([e1.x, e1.y, e1.z]),
        np.asarray([e2.x, e2.y, e2.z]),
        np.asarray([e3.x, e3.y, e3.z]),
    )


def get_camera_tform_fiducial(
    detection,
    json_path: bytes,
) -> np.ndarray:
    # get camera intrinsics
    with open(str(json_path), "r", encoding="UTF-8") as json_file:
        camera_dict = json.load(json_file)
    intrinsics = np.asarray(camera_dict["intrinsics"]).reshape((3, 3))

    points_3D = np.asarray(
        [[-73, -73, 0], [73, -73, 0], [73, 73, 0], [-73, 73, 0], [0, 0, 0]]
    ).astype(np.float32)
    points_2D = np.vstack([detection.corners, detection.center]).astype(np.float32)
    _, rvec, tvec, _ = cv2.solvePnPRansac(
        objectPoints=points_3D,
        imagePoints=points_2D,
        cameraMatrix=intrinsics,
        distCoeffs=np.zeros((4, 1)),  # assume no distortion
        flags=0,
        rvec=detection.homography,
    )

    # build 4x4 camera extrinsics matrix from rvec (to rot mat via rodrigues, and
    # tvec in meters
    extrinsics = np.zeros((4, 4))
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = tvec.squeeze() / 1000
    extrinsics[3, 3] = 1

    return extrinsics


def get_ground_tform_camera(json_path: bytes) -> np.ndarray:
    with open(str(json_path), "r", encoding="UTF-8") as file:
        dict_ = json.load(file)
    return np.asarray(dict_["cameraPoseARFrame"]).reshape(4, 4)


def get_corrective_matrix_camera() -> np.ndarray:
    # corrects for weird orientation of initial matrix of 3D Scanner App
    roll, pitch, yaw = 180, 0, 0
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    return rotation.as_matrix()


def get_corrective_matrix_fiducial() -> np.ndarray:
    # corrects for weird orientation of initial matrix of 3D Scanner App
    return np.asarray([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])


def calculate_ground_tform_fiducial(
    camera_tform_fiducial: np.ndarray,
    ground_tform_camera: np.ndarray,
    corrective_matrix_camera: np.ndarray,
    corrective_matrix_fiducial: np.ndarray,
) -> (np.ndarray, np.ndarray):
    corr_ground_tform_camera = ground_tform_camera.copy()
    corr_ground_tform_camera[:3, :3] = (
        corr_ground_tform_camera[:3, :3] @ corrective_matrix_camera
    )
    camera_tform_fiducial = corr_ground_tform_camera @ camera_tform_fiducial
    corr_camera_tform_fiducial = camera_tform_fiducial.copy()
    corr_camera_tform_fiducial[:3, :3] = (
        corr_camera_tform_fiducial[:3, :3] @ corrective_matrix_fiducial
    )
    return corr_ground_tform_camera, corr_camera_tform_fiducial


def main() -> None:
    # paths
    config = recursive_config.Config()
    directory_path = config.get_subpath("prescans")
    directory_path = os.path.join(
        str(directory_path), config["pre_scanned_graphs"]["high_res"]
    )
    jpg_json_paths, point_cloud_path = fetch_paths(directory_path)

    # take first image of scan
    tag_id = config["pre_scanned_graphs"]["base_fiducial_id"]
    best_frame_number, best_detection = get_best_detection(jpg_json_paths, tag_id)
    print(f"{best_frame_number=}")
    json_path = jpg_json_paths[best_frame_number]["json"]

    # calculate 4x4 rot + translation matrix ground_tform_fiducial
    camera_tform_fiducial = get_camera_tform_fiducial(best_detection, json_path)
    ground_tform_camera = get_ground_tform_camera(json_path)
    # corrective matrices are for correctly rotating the coordinate axes to wished
    # position
    corrective_matrix_camera = get_corrective_matrix_camera()
    corrective_matrix_fiducial = get_corrective_matrix_fiducial()
    ground_tform_camera, ground_tform_fiducial = calculate_ground_tform_fiducial(
        camera_tform_fiducial,
        ground_tform_camera,
        corrective_matrix_camera,
        corrective_matrix_fiducial,
    )

    # get point clouds
    scan = o3d.io.read_point_cloud(point_cloud_path)
    scan = add_coordinate_system(scan, (255, 0, 0))

    z, e1, e2, e3 = get_e_coordinates(ground_tform_camera)
    scan = add_coordinate_system(scan, (0, 255, 0), z, e1, e2, e3, False)

    z, e1, e2, e3 = get_e_coordinates(ground_tform_fiducial)
    scan = add_coordinate_system(scan, (0, 0, 255), z, e1, e2, e3, False)
    draw_cloud(scan)


if __name__ == "__main__":
    main()
