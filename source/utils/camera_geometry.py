import numpy as np

import open3d as o3d
from utils import vis


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
