import os

import numpy as np

import open3d as o3d
from utils import recursive_config
from utils.coordinates import Coordinates, get_circle_points
from utils.importer import PointCloud
from utils.docker_interfaces.mask3D_interface import get_coordinates_from_item


def get_top_n_entries(
    value_array: np.ndarray, coordinate_array: np.ndarray, n: int = 5
) -> np.ndarray:
    # Flatten the array and get the indices that would sort it in descending order
    flat_indices = np.argsort(-value_array.flatten())

    # Get the indices of the top n entries
    top_5_indices = np.unravel_index(flat_indices[:n], value_array.shape)

    # Get the corresponding values
    top_n_coordinates = coordinate_array[top_5_indices]
    return top_n_coordinates


def main() -> None:
    # CONSTANTS
    ITEM = "backpack"
    HEIGHT_THRESHOLD = -0.1
    BODY_HEIGHT = 0.45
    TARGET_COORDINATES = Coordinates((-0.5, 0, 0.1))
    # TARGET_COORDINATES = Coordinates((-0.9261271315789458, 0.7401371907894729, 0.1))
    N_BEST = 5
    LAMBDA = 0.5

    # paths
    config = recursive_config.Config()
    directory_path = config.get_subpath("aligned_point_clouds")
    pc_path = os.path.join(
        str(directory_path), config["pre_scanned_graphs"]["high_res"]
    )
    pc_path = os.path.join(pc_path, "scene.ply")

    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    # coordinates from item
    item_cloud, environment_cloud = get_coordinates_from_item(ITEM, mask_path, pc_path)
    target_coordinates = np.mean(np.asarray(item_cloud.points), axis=0)
    if TARGET_COORDINATES is not None:
        target_coordinates = TARGET_COORDINATES.get_as_ndarray()

    # delete floor from point cloud, so it doesn't interfere with the SDF
    points = np.asarray(environment_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > HEIGHT_THRESHOLD
    index = np.where(points_bool)[0]
    pc_no_ground = environment_cloud.select_by_index(index)

    # get points radiating outwards from target coordinate
    circle_points = get_circle_points(
        resolution=64,
        nr_circles=3,
        start_radius=0.75,
        end_radius=1,
        return_cartesian=True,
    )
    ## get center of radiating circle
    target_at_body_height = target_coordinates.copy()
    target_at_body_height[-1] = BODY_HEIGHT
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    ## add the radiating circle center to the points to elevate them
    circle_points = circle_points + target_at_body_height
    ## filter every point that is outside the scanned scene
    circle_points_bool = (min_points <= circle_points) & (circle_points <= max_points)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool]
    filtered_circle_points = filtered_circle_points.reshape((-1, 3))

    # transform point cloud to mesh to calculate SDF from
    ball_sizes = (0.02, 0.011, 0.005)
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    mesh_no_ground = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc_no_ground, radii=ball_sizes
    )
    mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_no_ground_legacy)

    # of the filtered points, cast ray from target to point to see if there are
    # collisions
    ray_directions = filtered_circle_points - target_coordinates
    rays_starts = np.tile(target_coordinates, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 3
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    circle_tensors = o3d.core.Tensor(
        filtered_circle_points, dtype=o3d.core.Dtype.Float32
    )

    # calculate the best body positions
    ## calculate SDF distances
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    ## calculate distance to target
    target_coordinates = target_coordinates.reshape((1, 1, 3))
    target_distances = filtered_circle_points - target_coordinates
    target_distances = target_distances.squeeze()
    target_distances = np.linalg.norm(target_distances, ord=2, axis=-1)
    ## get the top n coordinates
    values_to_maximize = distances - LAMBDA * target_distances
    top_n_coordinates = get_top_n_entries(
        values_to_maximize, filtered_circle_points, n=N_BEST
    )

    # draw the entries in the cloud
    drawable_geometries = [environment_cloud]
    for idx, coordinate in enumerate(top_n_coordinates, 1):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(coordinate)
        color = np.asarray([0, 1, 0]) * (idx / N_BEST)
        sphere.paint_uniform_color(color)
        drawable_geometries.append(sphere)

    o3d.visualization.draw_geometries(drawable_geometries)


if __name__ == "__main__":
    main()
