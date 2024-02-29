"""
All things video and imaging.

Some useful proto definitions:
ImageResponse (the full response sent by the robot):
https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#imageresponse
ImageSource (Intrinsics):
https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#imagesource
Image Capture (Extrinsics):
https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#image-pixelformat
PixelFormats: https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#image-pixelformat
"""

from __future__ import annotations

import os.path
from collections.abc import Iterable
from typing import Optional

import numpy as np

import apriltag
import cv2
import open3d as o3d
from bosdyn.api import image_pb2
from bosdyn.api.image_pb2 import ImageCapture, ImageResponse, ImageSource
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import (
    ImageClient,
    build_image_request,
    depth_image_to_pointcloud,
)
from bosdyn.client.world_object import WorldObjectClient
from robot_utils import frame_transformer as ft
from robot_utils.basic_movements import set_gripper
from robot_utils.frame_transformer import FrameTransformerSingleton
from scipy import ndimage
from scipy.interpolate import griddata
from utils import vis
from utils.coordinates import Pose3D
from utils.drawer_detection import BBox
from utils.importer import PointCloud, Vector3dVector
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.singletons import (
    ImageClientSingleton,
    RobotSingleton,
    WorldObjectClientSingleton,
)
from utils.vis import show_two_geometries_colored

frame_transformer = FrameTransformerSingleton()
image_client = ImageClientSingleton()
robot = RobotSingleton()
world_object_client = WorldObjectClientSingleton()

GRIPPER_DEPTH = "hand_depth"
GRIPPER_IMAGE_COLOR = "hand_color_image"
GRIPPER_IMAGE_GRAYSCALE = "hand_image"
GRIPPER_DEPTH_IN_COLOR = "hand_depth_in_hand_color_frame"
GRIPPER_COLOR_IN_DEPTH = "hand_color_in_hand_depth_frame"

ROTATION_ANGLE = {
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -90,  # -78,
    "frontright_fisheye_image": -90,  # -102,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
    GRIPPER_IMAGE_COLOR: 0,
    GRIPPER_IMAGE_GRAYSCALE: 0,
    GRIPPER_DEPTH_IN_COLOR: 0,
    GRIPPER_COLOR_IN_DEPTH: 0,
    "back_depth": 0,
    "frontleft_depth": -90,  # -78,
    "frontright_depth": -90,  # -102,
    "left_depth": 0,
    "right_depth": 180,
    "hand_depth": 0,
}

ALL_IMAGE_GREYSCALE_SOURCES = (
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "hand_image",
)

ALL_DEPTH_SOURCES = (
    "back_depth",
    "frontleft_depth",
    "frontright_depth",
    "hand_depth",
    "left_depth",
    "right_depth",
)


def get_pictures_from_sources(
    image_sources: Iterable[str],
    pixel_format: image_pb2.Image.PixelFormat,
    save_path: Optional[str] = None,
    auto_rotate: bool = True,
    vis_block: bool = False,
) -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Get picture from specified sources
    :param image_sources: iterable (list) of sensors from which readings should be taken
    :param pixel_format: kind of pixel format for the captures
    :param save_path: where to save the image (None -> not saved)
    :param auto_rotate: whether to auto rotate the image
    :param vis_block: whether to show the captured images before returning
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture
    """
    set_gripper(True)
    image_request = [
        build_image_request(source, pixel_format=pixel_format)
        for source in image_sources
    ]
    robot.logger.info("Sending image request.")
    image_responses = image_client.get_image(image_request)
    robot.logger.info("Received image response.")

    images = []
    for image_response in image_responses:
        num_bytes = 1  # Assume a default of 1 byte encodings.
        if (
            image_response.shot.image.pixel_format
            == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        ):
            dtype = np.uint16
            extension = "png"
        else:
            if (
                image_response.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_RGB_U8
            ):
                num_bytes = 3
            elif (
                image_response.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_RGBA_U8
            ):
                num_bytes = 4
            elif (
                image_response.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
            ):
                num_bytes = 1
            elif (
                image_response.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16
            ):
                num_bytes = 2
            dtype = np.uint8
            extension = "jpg"

        img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
        if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into a RGB rows X cols shape.
                img = img.reshape(
                    (
                        image_response.shot.image.rows,
                        image_response.shot.image.cols,
                        num_bytes,
                    )
                )
            except ValueError:
                robot.logger.info("Custom Raw Decode failed!")
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        if auto_rotate:
            img = ndimage.rotate(img, ROTATION_ANGLE[image_response.source.name])

        if save_path is not None:
            image_saved_path = image_response.source.name
            image_saved_path = image_saved_path.replace("/", "")
            img_save_path = os.path.join(save_path, f"{image_saved_path}.{extension}")
            cv2.imwrite(img_save_path + extension, img)

        if vis_block:
            vis.show_image(img, f"Image Source: {image_response.source.name}")

        # image returned
        images.append((img, image_response))
    return images


def get_rgb_pictures(
    image_sources: Iterable[str],
    auto_rotate: bool = True,
    vis_block: bool = False,
) -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Get rgb pictures of specified image sources.
    :param image_sources: iterable (list) of sensors from which readings should be taken
    :param auto_rotate: whether to auto rotate the image
    :param vis_block: whether to show the captured images before returning
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture
    """
    kwargs = {
        "image_sources": image_sources,
        "auto_rotate": auto_rotate,
        "vis_block": vis_block,
    }
    pixel_format = image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
    images = get_pictures_from_sources(pixel_format=pixel_format, **kwargs)
    return images


def get_greyscale_pictures(
    image_sources: Iterable[str],
    auto_rotate: bool = True,
    vis_block: bool = False,
) -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Get greyscale pictures of specified image sources.
    :param image_sources: iterable (list) of sensors from which readings should be taken
    :param auto_rotate: whether to auto rotate the image
    :param vis_block: whether to show the captured images before returning
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture
    """
    kwargs = {
        "image_sources": image_sources,
        "auto_rotate": auto_rotate,
        "vis_block": vis_block,
    }
    pixel_format = image_pb2.Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8
    images = get_pictures_from_sources(pixel_format=pixel_format, **kwargs)
    return images


def get_d_pictures(
    image_sources: Iterable[str],
    auto_rotate: bool = True,
    vis_block: bool = False,
) -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Get depth pictures of specified image sources.
    :param image_sources: iterable (list) of sensors from which readings should be taken
    :param auto_rotate: whether to auto rotate the image
    :param vis_block: whether to show the captured images before returning
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture
    """
    kwargs = {
        "image_sources": image_sources,
        "auto_rotate": auto_rotate,
        "vis_block": vis_block,
    }
    pixel_format = image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
    images = get_pictures_from_sources(pixel_format=pixel_format, **kwargs)
    return images


def get_camera_rgbd(
    in_frame: str = "image",
    cut_to_size: bool = True,
    auto_rotate: bool = True,
    vis_block: bool = False,
) -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Capture rgbd image from the gripper.
    :param in_frame: whether to capture depth in the image frame (frame="image") or image in the depth frame
    (frame="depth")
    :param cut_to_size: depth has a smaller FoV than the image, whether to cut image so size of depth FoV
    :param auto_rotate: whether to auto rotate the image
    :param vis_block: whether to visualize the rgbd image before returning
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture
    """
    if in_frame == "image":
        depth_sources = [GRIPPER_DEPTH_IN_COLOR]
        image_sources = [GRIPPER_IMAGE_COLOR]
    elif in_frame == "depth":
        depth_sources = [GRIPPER_DEPTH]
        image_sources = [GRIPPER_COLOR_IN_DEPTH]
    else:
        raise ValueError(f"in_frame must be in ['image', 'depth'], is {in_frame}")

    kwargs = {
        "auto_rotate": auto_rotate,
        "vis_block": vis_block,
    }
    # depth first
    depth_format = image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
    depth_images = get_pictures_from_sources(
        image_sources=depth_sources, pixel_format=depth_format, **kwargs
    )
    # color next
    color_format = image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
    color_images = get_pictures_from_sources(
        image_sources=image_sources, pixel_format=color_format, **kwargs
    )
    if in_frame == "image" and cut_to_size:
        # cut image to same FoV as depth
        depth_images = [
            (depth_image[0][:, 108:547], depth_image[1]) for depth_image in depth_images
        ]
        color_images = [
            (color_image[0][:, 108:547], color_image[1]) for color_image in color_images
        ]
    images = depth_images + color_images
    return images


def get_all_images_greyscale() -> list[(np.ndarray, image_pb2.ImageResponse)]:
    """
    Captures all available greyscale images.
    :return: list of (image as np array, bosdyn ImageResponse), where the first is the image as a numpy array, and the
    second is the ImageResponse object for every capture.
    """
    return get_greyscale_pictures(
        ALL_IMAGE_GREYSCALE_SOURCES,
        auto_rotate=False,
    )


def intrinsics_from_ImageSource(
    image_source: ImageSource, correct: bool = True
) -> np.ndarray:
    """
    Extract camera intrinsics from Image.source
    :param image_source: Image.source
    :param correct: TODO
    :return: (3, 3) np array of the camera intrinsics
    """
    cam_ints = image_source.pinhole.intrinsics
    # if cam_ints.skew != 0:
    #    print(UserWarning(f"Skew is not 0, it is {cam_ints.skew}!"))
    f, c = cam_ints.focal_length, cam_ints.principal_point
    return np.asarray(
        [
            [f.x, 0, 203 if correct else c.x],
            [0, f.y, c.y],
            [0, 0, 1],
        ]
    )


def camera_pose_from_ImageCapture(
    image_capture: ImageCapture,
    frame_relative_to: str = ODOM_FRAME_NAME,
) -> Pose3D:
    """
    Compute frame_tform_sensor.
    :param image_capture: Image.shot of bosdyn image client
    :param frame_relative_to: frame to which the transformation is relative to
    :return: frame_tform_sensor Pose3D
    """
    transforms_snapshot = image_capture.transforms_snapshot
    camera_frame_name = image_capture.frame_name_image_sensor
    camera_tform_frame_bd = get_a_tform_b(
        transforms_snapshot, camera_frame_name, frame_relative_to
    )
    camera_tform_frame = Pose3D.from_bosdyn_pose(camera_tform_frame_bd)
    return camera_tform_frame


def extrinsics_from_ImageCapture(
    image_capture: ImageCapture,
    relative_to_frame: str = BODY_FRAME_NAME,
) -> Pose3D:
    """
    Compute frame_tform_sensor.
    :param image_capture: Image.shot of bosdyn image client
    :param relative_to_frame: frame to which the transformation is relative to
    :return: frame_tform_sensor Pose3D
    """
    transforms_snapshot = image_capture.transforms_snapshot
    sensor_frame = image_capture.frame_name_image_sensor
    frame_tform_sensor = get_a_tform_b(
        transforms_snapshot, relative_to_frame, sensor_frame
    )
    return Pose3D.from_bosdyn_pose(frame_tform_sensor)


def build_surrounding_point_cloud() -> PointCloud:
    """
    Build point cloud of the environment surrounding the robot using all available depth cameras.
    :return: returns point cloud of environment.
    """
    set_gripper(True)
    dimage_tuples = get_d_pictures(ALL_DEPTH_SOURCES, auto_rotate=False)
    pcd_body = point_cloud_from_camera_captures(
        dimage_tuples, frame_relative_to=BODY_FRAME_NAME
    )
    return pcd_body


class NoFiducialDetectedError(Exception):
    pass


def localize_from_images(config: Config, vis_block: bool = False) -> str:
    """
    Localize the robot from camera images and depth scans of the surrounding environment.
    :param config:
    :param vis_block: whether to visualize the ICP part of the localization
    """
    """
    The idea is based on localization with fiducials and fine-tuning. Namely, we
    (1) first get an initial localization via the fiducials in the environment
    (2) then scan a point cloud of the surrounding environment
    (2) transform it to the fiducial frame based on its localization ("prediction")
    (3) load the pre-scanned point cloud as the "ground truth"
    (4) compare prediction and ground truth via ICP and save the adjusted localization
    """
    # create the apriltag (fiducial) detector
    tag_id = config["pre_scanned_graphs"]["base_fiducial_id"]
    options = apriltag.DetectorOptions(families="tag36h11", refine_pose=True)
    detector = apriltag.Detector(options)

    # set gripper to open to capture images from that camera as well
    set_gripper(True)

    # get images from all robot cameras
    global image_client
    image_client.set_instance(robot.ensure_client(ImageClient.default_service_name))

    image_tuples = get_all_images_greyscale()

    best_fit = -1
    best_frame_idx = -1
    best_detection = None

    # get image that best shows the apriltag
    for frame_idx, (image, _) in enumerate(image_tuples):
        detections = detector.detect(image)
        for detection in detections:
            if detection.tag_id != tag_id:
                continue
            if detection.goodness > best_fit:
                best_fit = detection.decision_margin
                best_frame_idx = frame_idx
                best_detection = detection

    if best_detection is None:
        raise NoFiducialDetectedError()

    # use that image to compute the current pose relative to it (body_tform_fiducial)
    _, best_response = image_tuples[best_frame_idx]
    intrinsics = intrinsics_from_ImageSource(best_response.source)
    body_tform_camera = extrinsics_from_ImageCapture(
        best_response.shot, BODY_FRAME_NAME
    ).as_matrix()
    robot.logger.info(
        f"{best_frame_idx=}, name={image_tuples[best_frame_idx][1].shot.frame_name_image_sensor}"
    )

    # extract camera_tform_fiducial
    points_3D = np.asarray(
        [[-73, -73, 0], [73, -73, 0], [73, 73, 0], [-73, 73, 0], [0, 0, 0]]
    ).astype(np.float32)
    points_2D = np.vstack([best_detection.corners, best_detection.center]).astype(
        np.float32
    )
    _, rvec, tvec, _ = cv2.solvePnPRansac(
        objectPoints=points_3D,
        imagePoints=points_2D,
        cameraMatrix=intrinsics,
        distCoeffs=np.zeros((4, 1)),  # assume no distortion
        flags=0,
        rvec=best_detection.homography,
    )

    camera_tform_fiducial = np.eye(4)
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    camera_tform_fiducial[:3, :3] = rotation_matrix
    camera_tform_fiducial[:3, 3] = tvec.squeeze() / 1000

    body_tform_fiducial = body_tform_camera @ camera_tform_fiducial

    # create frame frame_transformer object
    global world_object_client
    world_object_client.set_instance(
        robot.ensure_client(WorldObjectClient.default_service_name)
    )
    ft.FrameTransformer()
    frame_name = ft.VISUAL_SEED_FRAME_NAME

    # correct for turn
    body_tform_fiducial[:3, :3] = body_tform_fiducial[:3, :3] @ np.asarray(
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    )
    fiducial_tform_body = np.linalg.inv(body_tform_fiducial)

    # build the point cloud of the surrounding environment
    pcd_body = build_surrounding_point_cloud()
    # transform it into frame relative to fiducial (ground truth is centered on fiducial)
    # this is the "prediction" of the environment based on the localization w.r.t. the fiducial
    pcd_fiducial = pcd_body.transform(fiducial_tform_body)

    # get the ground truth point cloud (pre-scanned)
    base_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    scan_path = os.path.join(base_path, ending, "scene.ply")
    pcd_ground = o3d.io.read_point_cloud(scan_path)

    if vis_block:
        show_two_geometries_colored(pcd_fiducial, pcd_ground)

    # get offset between prediction and ground truth
    ground_tform_fiducial = icp(
        pcd_ground, pcd_fiducial, threshold=0.05, max_iteration=200
    )
    ground_tform_body = ground_tform_fiducial @ fiducial_tform_body

    if vis_block:
        pcd_new_ground = pcd_fiducial.transform(ground_tform_fiducial)
        show_two_geometries_colored(pcd_new_ground, pcd_ground)

    # compute transformation relative to odom to add to frame_transformer
    body_tform_odom = frame_transformer.transform_matrix(
        ODOM_FRAME_NAME, BODY_FRAME_NAME
    )
    ground_tform_odom = ground_tform_body @ body_tform_odom

    frame_transformer.add_frame_tform_vision(frame_name, ground_tform_odom)

    set_gripper(False)
    return frame_name


def relocalize(
    config: Config,
    frame_name: str,
    vis_block: bool = False,
) -> None:
    """
    Given an existing location, update the localization based on the surrounding point cloud.
    :param config:
    :param frame_name: frame name of seed frame
    :param vis_block: whether to visualize the ICP part of the process
    """
    """
    The idea is very similar to the initial localization. Namely, we
    (1) first scan a point cloud of the surrounding environment
    (2) transform it to the seed frame based on the current localization ("prediction")
    (3) load the pre-scanned point cloud as the "ground truth"
    (4) compare prediction and ground truth via ICP and save the adjusted localization
    """

    # first build the surrounding point cloud in the body frame
    dyn_pcd_body = build_surrounding_point_cloud()
    frame_tform_body = frame_transformer.transform_matrix(BODY_FRAME_NAME, frame_name)
    # transform point cloud to the seed frame
    # this is based on the current localization of the robot
    dyn_pcd_frame = dyn_pcd_body.transform(frame_tform_body)

    # get the pre-scanned point cloud as ground truth
    base_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    scan_path = os.path.join(base_path, ending, "scene.ply")
    pcd_ground = o3d.io.read_point_cloud(scan_path)

    if vis_block:
        show_two_geometries_colored(dyn_pcd_frame, pcd_ground)

    # align the ground truth (pcd_ground) and the one from current localization
    ground_tform_frame = icp(pcd_ground, dyn_pcd_frame)
    ground_tform_body = ground_tform_frame @ frame_tform_body

    if vis_block:
        dyn_pcd_ground = dyn_pcd_frame.transform(ground_tform_frame)
        show_two_geometries_colored(dyn_pcd_ground, pcd_ground)

    # get the new localization in the odom frame to add to the frame frame_transformer
    body_tform_odom = frame_transformer.transform_matrix(
        ODOM_FRAME_NAME, BODY_FRAME_NAME
    )
    ground_tform_odom = ground_tform_body @ body_tform_odom

    frame_transformer.add_frame_tform_vision(
        frame_name, ground_tform_odom, overwrite=True
    )

    set_gripper(False)


def point_cloud_from_camera_captures(
    depth_images: list[(np.ndarray, image_pb2.ImageResponse)],
    frame_relative_to: str = BODY_FRAME_NAME,
) -> PointCloud:
    """
    Given a list of (depth_image, ImageResponse), compute the combined point cloud relative to the specified frame.
    :param depth_images: list of (depth_image, ImageResponse)
    :param frame_relative_to: frame relative to which the point cloud will be returned
    :return: combined point cloud
    """
    fused_point_clouds = PointCloud()
    for _, image_response in depth_images:
        pcd_camera_np = depth_image_to_pointcloud(image_response)
        pcd_camera = PointCloud()
        pcd_camera.points = Vector3dVector(pcd_camera_np)
        camera_tform_frame = camera_pose_from_ImageCapture(
            image_response.shot, frame_relative_to
        )
        frame_tform_camera = camera_tform_frame.inverse()
        pcd_frame = pcd_camera.transform(frame_tform_camera.as_matrix())
        fused_point_clouds = fused_point_clouds + pcd_frame
    return fused_point_clouds


def frame_coordinate_from_depth_image(
    depth_image: np.ndarray,
    depth_image_response: ImageResponse,
    pixel_coordinatess: np.ndarray,
    frame_name: str,
) -> np.ndarray:
    """
    Compute a 3D coordinate from a depth image and pixel coordinate.
    :param depth_image: depth image, shape (H, W, 1)
    :param depth_image_response: associated ImageResponse
    :param pixel_coordinatess: coordinates of the pixel to get the 3D coordinate of (format: height, width)
    :param frame_name: frame relative to which to express the 3D coordinate
    """
    assert pixel_coordinatess.ndim == 2 and pixel_coordinatess.shape[-1] == 2
    pixel_coordinates_flipped = np.flip(pixel_coordinatess, axis=-1)

    # Get the valid depth measurements
    depth_image = depth_image.squeeze()
    valid_coords = np.argwhere(depth_image != 0)
    valid_depths = depth_image[depth_image != 0]
    target_depth = griddata(
        valid_coords, valid_depths, pixel_coordinates_flipped, method="cubic"
    )
    target_depth = target_depth / depth_image_response.source.depth_scale
    print(f"{target_depth=}")

    # prepare intrinsics, extrinsics
    intrinsics = intrinsics_from_ImageSource(depth_image_response.source)
    camera_tform_odom = camera_pose_from_ImageCapture(
        depth_image_response.shot, ODOM_FRAME_NAME
    )
    odom_tform_camera = camera_tform_odom.inverse(inplace=False).as_matrix()
    frame_tform_odom = frame_transformer.transform_matrix(ODOM_FRAME_NAME, frame_name)
    frame_tform_camera = frame_tform_odom @ odom_tform_camera

    # calculate the 3d coordinates in the camera frame
    ones = np.ones((pixel_coordinatess.shape[0], 1))
    pixel_coords_hom = np.concatenate((pixel_coordinatess, ones), axis=1)
    coords_camera_on_plane = pixel_coords_hom @ np.linalg.inv(intrinsics).T
    coords_camera_norm = (
        coords_camera_on_plane / coords_camera_on_plane[:, -1, np.newaxis]
    )
    coords_camera = coords_camera_norm * target_depth[:, np.newaxis]

    coords_camera_hom = np.concatenate((coords_camera, ones.copy()), axis=1)
    coords_frame = coords_camera_hom @ frame_tform_camera.T
    coords_frame_norm = coords_frame[:, :-1] / coords_frame[:, -1, np.newaxis]
    return coords_frame_norm


def project_3D_to_2D(
    depth_image_response: (np.ndarray, ImageResponse),
    coordinates_in_frame: np.ndarray,
    frame_name: str,
) -> np.ndarray:
    depth_image, depth_response = depth_image_response
    camera_tform_body = camera_pose_from_ImageCapture(
        depth_response.shot, BODY_FRAME_NAME
    ).as_matrix()
    if frame_name is None:
        frame_tform_body = camera_tform_body.copy()
    else:
        frame_tform_body = frame_transformer.transform_matrix(
            BODY_FRAME_NAME, frame_name
        )
    body_tform_frame = np.linalg.inv(frame_tform_body)
    camera_tform_frame = camera_tform_body @ body_tform_frame
    intrinsics = intrinsics_from_ImageSource(depth_response.source)

    # transform point cloud into camera frame
    ones = np.ones((coordinates_in_frame.shape[0], 1))
    pcd_frame_hom = np.concatenate((coordinates_in_frame, ones), axis=1)
    pcd_camera_hom = pcd_frame_hom @ camera_tform_frame.T
    pcd_camera = pcd_camera_hom[..., :3] / pcd_camera_hom[..., -1, np.newaxis]
    pcd_2d_hom = pcd_camera @ intrinsics.T
    pcd_2d = pcd_2d_hom[:, :2] / pcd_2d_hom[:, -1, np.newaxis]
    return pcd_2d


def select_points_from_bounding_box(
    depth_image_response: (np.ndarray, ImageResponse),
    bboxes: list[BBox],
    frame_name: str,
    vis_block: bool = False,
) -> (np.ndarray, np.ndarray):
    """
    Given a depth response, and a couple bounding boxes, compute (1) the point cloud in the given frame_name, and return
    (2) masks that select the points within each bounding box
    """
    # get necessary prerequisites
    depth_image, depth_response = depth_image_response
    pcd_body = point_cloud_from_camera_captures([depth_image_response])
    pcd_body = np.array(pcd_body.points)
    camera_tform_body = camera_pose_from_ImageCapture(
        depth_response.shot, BODY_FRAME_NAME
    ).as_matrix()
    if frame_name is None:
        frame_tform_body = camera_tform_body.copy()
    else:
        frame_tform_body = frame_transformer.transform_matrix(
            BODY_FRAME_NAME, frame_name
        )
    intrinsics = intrinsics_from_ImageSource(depth_response.source)

    # transform point cloud into camera frame
    ones = np.ones((pcd_body.shape[0], 1))
    pcd_body_hom = np.concatenate((pcd_body, ones), axis=1)
    pcd_camera_hom = pcd_body_hom @ camera_tform_body.T
    pcd_camera = pcd_camera_hom[..., :3] / pcd_camera_hom[..., -1, np.newaxis]
    pcd_2d_hom = pcd_camera @ intrinsics.T
    pcd_2d = pcd_2d_hom[:, :2] / pcd_2d_hom[:, -1, np.newaxis]

    # get masks for all bounding boxes
    bbox_masks = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = [int(val) for val in bbox]
        x_mask = (xmin <= pcd_2d[:, 0]) & (pcd_2d[:, 0] <= xmax)
        y_mask = (ymin <= pcd_2d[:, 1]) & (pcd_2d[:, 1] <= ymax)
        mask = x_mask & y_mask
        bbox_masks.append(mask)
    bbox_masks_np = np.stack(bbox_masks, axis=0)

    # transform points into reference frame
    pcd_frame_hom = pcd_body_hom @ frame_tform_body.T
    pcd_frame = pcd_frame_hom[..., :3] / pcd_frame_hom[..., -1, np.newaxis]

    if vis_block:
        pcd_vis = PointCloud()
        pcd_vis.points = Vector3dVector(pcd_frame)
        for bbox_mask in bbox_masks_np:
            pcd_in = pcd_vis.select_by_index(np.where(bbox_mask)[0])
            pcd_out = pcd_vis.select_by_index(np.where(~bbox_mask)[0])
            pcd_in.paint_uniform_color([0, 1, 0])
            pcd_out.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([pcd_in, pcd_out])

    return pcd_frame, bbox_masks_np
