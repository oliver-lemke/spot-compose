from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

import apriltag
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from utils import recursive_config
from utils.importer import PointCloud
from utils.point_clouds import add_coordinate_system

SAVE_OPENMASK3D = True
os.environ["DISPLAY"] = "localhost:11.0"


def fetch_paths(
    directory_path: bytes | str,
) -> tuple[dict[int, dict[str, str]], str, str]:
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

    mesh_path = os.path.join(str(directory_path), "export_refined.obj")
    pcd_path = os.path.join(str(directory_path), "pcd.ply")
    assert os.path.exists(mesh_path), f"Mesh file does not exists at {mesh_path}!"

    return jpg_json_paths, mesh_path, pcd_path


def get_best_detection(
    jpg_json_paths, tag_id: int, set_detection_nr: int | None = None
):
    options = apriltag.DetectorOptions(families="tag36h11", refine_pose=True)
    detector = apriltag.Detector(options)

    if set_detection_nr is not None:
        jpg_json_paths = {set_detection_nr: jpg_json_paths[set_detection_nr]}

    best_fit = -1
    best_frame_number = -1
    best_detection = None

    for frame_nr, jpg_json_dict in tqdm(jpg_json_paths.items(), "Checking detections"):
        # get data (image and camera intrinsics)
        jpg_path = jpg_json_dict["jpg"]
        img = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)

        # raw image often runs into "warning: too many borders in contour_detect (max of 32767!)"
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # bad read here
        detections = detector.detect(img)
        for detection in detections:
            if detection.tag_id != tag_id:
                continue
            if detection.decision_margin > best_fit:
                best_fit = detection.decision_margin
                best_frame_number = frame_nr
                best_detection = detection
                break

    print(f"Found best detection at #{best_frame_number}!")
    return best_frame_number, best_detection


def get_camera_tform_fiducial(
    detection,
    json_path: bytes,
) -> np.ndarray:
    # get camera intrinsics
    intrinsics = get_camera_intrinsics(json_path)

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


def get_camera_intrinsics(json_path: bytes) -> np.ndarray:
    with open(str(json_path), "r", encoding="UTF-8") as json_file:
        camera_dict = json.load(json_file)
    return np.asarray(camera_dict["intrinsics"]).reshape((3, 3))


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
) -> np.ndarray:
    corr_ground_tform_camera = ground_tform_camera.copy()
    corr_ground_tform_camera[:3, :3] = (
        corr_ground_tform_camera[:3, :3] @ corrective_matrix_camera
    )
    camera_tform_fiducial = corr_ground_tform_camera @ camera_tform_fiducial
    corr_camera_tform_fiducial = camera_tform_fiducial.copy()
    corr_camera_tform_fiducial[:3, :3] = (
        corr_camera_tform_fiducial[:3, :3] @ corrective_matrix_fiducial
    )
    return corr_camera_tform_fiducial


def correct_to_upright(ground_tform_fiducial: np.ndarray) -> np.ndarray:
    """given a rotation matrix, corrects it in such a way that the z axis points in the same direction as the up axis always.

    Args:
        ground_tform_fiducial (np.ndarray): ground_tform_fiducial rotation matrix_

    Returns:
        np.ndarray: _description_
    """
    zero = np.asarray([0, 0, 0, 1])
    ez = np.asarray([0, 0, 1, 1])

    ground_ez = ground_tform_fiducial @ ez
    ground_ez = ground_ez[:3] / ground_ez[3]
    ground_zero = ground_tform_fiducial @ zero
    ground_zero = ground_zero[:3] / ground_zero[3]

    ground_ez = ground_ez - ground_zero

    roll = np.sin(ground_ez[0]) * np.sign(ground_ez[0])
    pitch = np.sin(ground_ez[2]) * np.sign(ground_ez[2])
    yaw = 0.0
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    corr_ground_tform_fiducial = ground_tform_fiducial.copy()
    corr_ground_tform_fiducial[:3, :3] = corr_ground_tform_fiducial[:3, :3] @ rotation
    return corr_ground_tform_fiducial


def cuda_to_tensor_pointcloud(cuda_pc):
    # Convert CUDA-based PointCloud to legacy PointCloud
    points = np.asarray(cuda_pc.points, dtype=np.float32)
    legacy_pc = o3d.t.geometry.PointCloud(points)
    return legacy_pc


def render_depth(mesh, camera):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1440, visible=False)
    vis.add_geometry(mesh)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera, True)

    # Capture depth buffer and create depth image
    depth = np.asarray(vis.capture_depth_float_buffer(True)) * 1000.0
    depth_scaled = depth.astype(np.uint16)
    image = np.asarray(vis.capture_screen_float_buffer(True))

    # Cleanup visualizer and return depth image
    vis.destroy_window()
    return depth_scaled, image


def save_ndarray(path: str, array: np.ndarray) -> None:
    save_string = "\n".join([" ".join(map(str, row)) for row in array])
    with open(path, "w", encoding="UTF-8") as f:
        f.write(save_string)


def main() -> None:
    # paths
    config = recursive_config.Config()
    directory_path = config.get_subpath("prescans")
    directory_path = os.path.join(
        str(directory_path), config["pre_scanned_graphs"]["high_res"]
    )
    jpg_json_paths, mesh_path, pcd_path = fetch_paths(directory_path)

    # take first image of scan_ground
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
    ground_tform_fiducial = calculate_ground_tform_fiducial(
        camera_tform_fiducial,
        ground_tform_camera,
        corrective_matrix_camera,
        corrective_matrix_fiducial,
    )
    ground_tform_fiducial = correct_to_upright(ground_tform_fiducial)
    fiducial_tform_ground = np.linalg.inv(ground_tform_fiducial)

    # get point clouds
    mesh_ground = o3d.io.read_triangle_mesh(mesh_path, True)
    scan_ground = o3d.io.read_point_cloud(pcd_path)

    scan_fiducial = copy.deepcopy(scan_ground).transform(fiducial_tform_ground)
    mesh_fiducial = copy.deepcopy(mesh_ground).transform(fiducial_tform_ground)
    scan_vis = add_coordinate_system(
        scan_fiducial, (0, 255, 0), np.asarray((0, 0, 0)), size=2
    )
    o3d.visualization.draw_geometries([scan_vis])

    # POINT CLOUD HAS BEEN TRANSFORMED with icp_tform_ground
    # now we create the scene folder structure for openmask3d
    # https://github.com/OpenMask3D/openmask3d
    save_path = config.get_subpath("aligned_point_clouds")
    save_path = os.path.join(str(save_path), config["pre_scanned_graphs"]["high_res"])
    pose_save_path = os.path.join(save_path, "pose")
    ground_tform_camera_save_path = os.path.join(
        pose_save_path, "ground_tform_camera.txt"
    )
    fiducial_tform_ground_save_path = os.path.join(
        pose_save_path, "fiducial_tform_ground.txt"
    )
    color_save_path = os.path.join(save_path, "color")
    depth_save_path = os.path.join(save_path, "depth")
    intrinsic_save_path = os.path.join(save_path, "intrinsic")
    cloud_save_path = os.path.join(save_path, "scene.ply")
    mesh_save_path = os.path.join(save_path, "mesh.obj")
    for current_path in (
        save_path,
        pose_save_path,
        color_save_path,
        depth_save_path,
        intrinsic_save_path,
    ):
        os.makedirs(current_path, exist_ok=False)

    intrinsic = None
    for frame_nr, jpg_json_dict in tqdm(jpg_json_paths.items(), "Saving renders"):
        # get data (image and camera intrinsics)
        jpg_path = jpg_json_dict["jpg"]
        jpg = cv2.imread(jpg_path)
        json_path = jpg_json_dict["json"]
        ground_tform_camera = get_ground_tform_camera(json_path)
        ground_tform_camera[:3, :3] = (
            ground_tform_camera[:3, :3] @ corrective_matrix_camera
        )

        fiducial_tform_camera = fiducial_tform_ground @ ground_tform_camera
        camera_tform_fiducial = np.linalg.inv(fiducial_tform_camera)

        pose_path = os.path.join(pose_save_path, f"{frame_nr}.txt")
        save_ndarray(pose_path, np.linalg.inv(camera_tform_fiducial))

        intrinsics = get_camera_intrinsics(json_path)
        intrinsic = intrinsics

        if SAVE_OPENMASK3D:
            height, width = jpg.shape[:2]
            camera = o3d.camera.PinholeCameraParameters()
            camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height, intrinsic_matrix=intrinsics
            )
            camera.extrinsic = camera_tform_fiducial
            depth, _ = render_depth(mesh_fiducial, camera)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 255
            image_rgb = jpg

            color_path = os.path.join(color_save_path, f"{frame_nr}.jpg")
            depth_path = os.path.join(depth_save_path, f"{frame_nr}.png")

            cv2.imwrite(color_path, cv2.resize(image_rgb, (640, 480)))
            cv2.imwrite(depth_path, cv2.resize(depth, (640, 480)))

    # need 4x4 matrix for intrinsics for openmask
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsic
    save_ndarray(
        os.path.join(intrinsic_save_path, "intrinsic_color.txt"), intrinsics_4x4
    )
    save_ndarray(ground_tform_camera_save_path, ground_tform_camera)
    save_ndarray(fiducial_tform_ground_save_path, fiducial_tform_ground)
    o3d.io.write_point_cloud(cloud_save_path, scan_fiducial)
    o3d.io.write_triangle_mesh(mesh_save_path, mesh_fiducial)


if __name__ == "__main__":
    main()
    sys.exit(0)
