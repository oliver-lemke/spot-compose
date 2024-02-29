import numpy as np

import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RANSACRegressor
from utils import vis


def plane_fitting(
    pcd: np.ndarray,
    threshold: float = 0.01,
    min_samples: int = 3,
    vis_block: bool = False,
) -> np.ndarray:
    assert pcd.shape[-1] == 3

    X = pcd[:, 1:]  # y and z coordinates
    y = pcd[:, 0]  # x coordinates
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min_samples,
        residual_threshold=threshold,
        max_trials=100,
    )
    ransac.fit(X, y)

    if vis_block:
        vis.show_point_cloud_in_out(pcd, ransac.inlier_mask_)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    normal = np.array([1, -a, -b])
    normal_normalized = normal / np.linalg.norm(normal)
    return normal_normalized


def fit_plane_pca_ransac(
    pcd: np.ndarray,
    threshold: float = 0.01,
    min_samples: int = 3,
    n_iterations: int = 1000,
    vis_block: bool = False,
):
    best_inliers_count = -1
    best_inliers = None
    best_normal_vector = None
    best_D = None

    for _ in range(n_iterations):
        # Step 1: Random sampling
        if pcd.shape[0] < min_samples:
            raise ValueError("Number of points is less than min_samples.")

        sampled_indices = np.random.choice(
            pcd.shape[0], size=min_samples, replace=False
        )
        sampled_points = pcd[sampled_indices]

        # Center the sampled points before PCA
        sampled_points_centered = sampled_points - np.mean(sampled_points, axis=0)

        # Step 2: Apply PCA
        pca = PCA(n_components=2).fit(sampled_points_centered)
        normal_vector = np.cross(pca.components_[0], pca.components_[1])
        point_on_plane = np.mean(sampled_points, axis=0)
        D = -np.dot(normal_vector, point_on_plane)

        # Step 3: Identify inliers
        distances = np.abs(np.dot(pcd, normal_vector) + D) / np.linalg.norm(
            normal_vector
        )
        inliers = distances < threshold
        inliers_count = np.sum(inliers)

        # Step 4: Evaluate model
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers = inliers
            best_normal_vector = normal_vector / np.linalg.norm(
                normal_vector
            )  # Normalize the normal vector
            best_D = D

    if vis_block:
        vis.show_point_cloud_in_out(pcd, best_inliers)

    return best_normal_vector


def plane_fitting_open3d(
    pcd: np.ndarray,
    threshold: float = 0.01,
    min_samples: int = 3,
    n_iterations: int = 1000,
    vis_block: bool = False,
):
    points = pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Use RANSAC to find the plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=threshold, ransac_n=min_samples, num_iterations=n_iterations
    )

    if vis_block:
        in_mask = np.zeros((points.shape[0],))
        in_mask[inliers] = 1.0
        in_mask = in_mask.astype(bool)
        vis.show_point_cloud_in_out(points, in_mask)

    a, b, c, d = plane_model
    normal_vector = np.array([a, b, c])

    # Optionally, you might want to normalize this vector to unit length
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector_normalized
