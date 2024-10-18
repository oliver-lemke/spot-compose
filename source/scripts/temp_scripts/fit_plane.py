from __future__ import annotations

import json
import os

import numpy as np

import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LinearRegression, RANSACRegressor
from utils.drawer_detection import predict_darknet as drawer_predict
from utils.recursive_config import Config


def filter_detections(detections: list[tuple[str, float, list[float]]], name: str):
    return [detection for detection in detections if detection[0] == name]


def select_points_in_bbox(points, intrinsics, extrinsics, bbox):
    # Convert points to homogeneous coordinates (add a column of ones)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply the extrinsic matrix (camera pose) to transform points
    points_camera = points_homogeneous @ np.linalg.inv(extrinsics.T)
    points_camera = points_camera[:, :3]
    # Project points onto the image plane using the intrinsic matrix
    points_img_hom = points_camera @ intrinsics.T
    # Normalize by the third (z) component to get pixel coordinates
    front_of_cam_filter = points_img_hom[:, -1] > 0
    points_img = points_img_hom[:, :2] / points_img_hom[:, 2, np.newaxis]

    x_min, y_min, x_max, y_max = bbox
    bbox_min = np.array([x_min, y_min])
    bbox_max = np.array([x_max, y_max])

    in_bbox_filter = np.all((points_img >= bbox_min) & (points_img <= bbox_max), axis=1)

    full_filter = front_of_cam_filter & in_bbox_filter
    return points_camera, points_img, full_filter


def main():
    # set up paths
    config = Config()
    data_path = config.get_subpath("resources")
    base_path = os.path.join(data_path, "prescans", "24-02-22")
    frame_nr = 147
    frame_nr_str = str(frame_nr).zfill(5)
    image_name = f"frame_{frame_nr_str}.jpg"
    json_name = f"frame_{frame_nr_str}.json"
    image_path = os.path.join(base_path, image_name)
    json_path = os.path.join(base_path, json_name)
    mesh_path = os.path.join(base_path, "textured_output.obj")

    # load image, camera settings, mesh, pcd
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(json_path, "r") as file:
        camera_dict = json.load(file)
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=100_000, use_triangle_normal=True
    )

    detections = drawer_predict(image, config, vis_block=True)
    drawer_detections = filter_detections(detections, "cabinet door")
    x_center, y_center, w, h = drawer_detections[0][2]
    x_min, y_min = x_center - w / 2, y_center - h / 2
    x_max, y_max = x_center + w / 2, y_center + h / 2
    bbox = [x_min, y_min, x_max, y_max]

    intrinsics = np.array(camera_dict["intrinsics"]).reshape((3, 3))
    extrinsics = np.array(camera_dict["cameraPoseARFrame"]).reshape((4, 4))
    correct_matrix = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    # correct_matrix = np.eye(3)
    extrinsics[:3, :3] = extrinsics[:3, :3] @ correct_matrix

    points = np.array(pcd.points)
    points_camera, _, mask = select_points_in_bbox(points, intrinsics, extrinsics, bbox)

    distance_min = 0
    distance_max = 0.65
    point_distances = np.linalg.norm(points_camera, 2, axis=1)
    distance_mask = (point_distances >= distance_min) & (
        point_distances <= distance_max
    )
    full_mask = distance_mask & mask

    pcd_in = pcd.select_by_index(np.where(full_mask)[0])
    pcd_out = pcd.select_by_index(np.where(~full_mask)[0])
    pcd_in.paint_uniform_color([0, 1, 0])
    pcd_out.paint_uniform_color([1, 0, 0])

    # plane fitting
    points_in = points[full_mask]
    X = points_in[:, :2]  # x and y coordinates
    y = points_in[:, 2]  # z coordinates
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=3,
        residual_threshold=0.02,
        max_trials=100,
    )
    ransac.fit(X, y)

    # Extract the inlier mask and coefficients
    inlier_mask = ransac.inlier_mask_

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # Generate a grid of x, y points
    x = np.linspace(-1, 1, 100)  # Adjust these bounds as needed
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)

    # Calculate corresponding z values
    zz = a * xx + b * yy + c

    # Flatten the arrays and create a combined Nx3 array for Open3D
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    zs = np.linspace(0, 1, 100) - c
    xys = np.ones((100, 2)) * np.array((a, b))
    extra_points = np.concatenate((xys, zs[:, np.newaxis]), axis=1)
    points = np.concatenate((points, extra_points), axis=0)

    # Create a point cloud object
    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(points)
    plane.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd_in, pcd_out])
    o3d.visualization.draw_geometries([pcd_in, pcd_out, plane])


def _test():
    # Camera intrinsic parameters
    fx, fy = 1000, 1000  # Focal length
    cx, cy = 500, 500  # Principal point
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Camera pose (identity matrix for this example)
    extrinsics = np.eye(4)

    # Bounding box in image space
    bbox = [400, 250, 600, 750]  # [x_min, y_min, x_max, y_max]

    # Points in world coordinates (some inside the frustum, some outside)
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.ones(X.shape) * 10
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()

    # Convert numpy arrays to Open3D format and assign to the point cloud
    points = np.vstack((x_flat, y_flat, z_flat)).T
    pcd.points = o3d.utility.Vector3dVector(points)

    points_img, mask = select_points_in_bbox(points, intrinsics, extrinsics, bbox)
    print(mask.sum())
    pcd_filt = pcd.select_by_index(np.where(mask)[0])

    sphere = o3d.geometry.TriangleMesh.create_sphere(5)
    o3d.visualization.draw_geometries([sphere, pcd_filt])


if __name__ == "__main__":
    main()
