# pylint: disable-all
from __future__ import annotations

import math
import os

import numpy as np

import open3d as o3d
from matplotlib import pyplot as plt
from utils import recursive_config
from utils.coordinates import get_uniform_sphere_directions
from utils.docker_interfaces.mask3D_interface import get_coordinates_from_item


def steps_to_change_from_right(rays: np.ndarray) -> np.ndarray:
    """
    Calculate - at each position -  the number of steps it takes to get to the next
    value (to the left) that is different to the value at the current position
    :param rays: boolean array
    :return: see above
    """
    change = np.abs(np.diff(rays, axis=-1, prepend=0)).astype(bool)
    counting_vector = np.arange(rays.shape[-1])
    counting_vector = np.tile(counting_vector, (*rays.shape[:-1], 1))
    index_vector = np.zeros(rays.shape)
    index_vector[change] = counting_vector[change]
    extended_array = np.maximum.accumulate(index_vector, axis=-1)
    return counting_vector - extended_array


def steps_to_change(rays):
    from_right = steps_to_change_from_right(rays)
    from_left = steps_to_change_from_right(np.flip(rays, axis=-1))
    from_left = np.flip(from_left, axis=-1)
    min_from_both = np.minimum(from_right, from_left)
    return min_from_both


def create_quadratic_matrix(size, sigma):
    indices = np.arange(size)
    matrix = np.exp(-((indices - indices[:, np.newaxis]) ** 2) / (2 * sigma**2))
    return matrix


def double_rays(distances: np.ndarray, angles: np.ndarray) -> (np.ndarray, np.ndarray):
    extra_angles = angles + np.asarray([0, math.tau])
    doubled_angles = np.concatenate([angles, extra_angles])
    doubles_dists = np.concatenate([distances, distances])
    return doubles_dists, doubled_angles


def repeat_scene(
    angles: np.ndarray, values: np.ndarray, below: bool = False
) -> (np.ndarray, np.ndarray):
    """
    We repeat the scene to account for spherical coordinates (2pi and 0 are the same)
    For that we repeat the scene once on the right, then mirrored above and
    (potentially) below.
    :param angles: spherical angles at which rays were cast, shape (n, 2n, 3)
    :param values: values for those angles
    :param below: whether to repeat the scene below as well
    :return: repeated scene, shape (2n, 4n, 3) or (3n, 4n, 3) depending on "below"
    """
    # vector format is (theta, phi)
    # right extension -> increase phi
    right_angles = angles.copy()
    right_angles[..., 1] = right_angles[..., 1] + math.tau
    right_extended_angles = np.concatenate([angles, right_angles], axis=1)
    right_extended_values = np.tile(values, (1, 2))

    # top extension -> increase theta
    top_angles = right_extended_angles.copy()
    top_angles[..., 0] = top_angles[..., 0] + math.pi
    top_extended_angles = np.concatenate([top_angles, right_extended_angles], axis=0)
    top_values = right_extended_values.copy()
    ## we have to flip both ways, because by extending theta we inherently flip
    top_values = np.flip(top_values)
    top_extended_values = np.concatenate([top_values, right_extended_values], axis=0)

    if not below:
        return top_extended_angles, top_extended_values

    # bot extension -> increase theta
    bot_angles = right_extended_angles.copy()
    bot_angles[..., 0] = bot_angles[..., 0] - math.pi
    bot_extended_angles = np.concatenate([top_extended_angles, bot_angles], axis=0)
    bot_values = right_extended_values.copy()
    ## we have to flip both ways, because by extending theta we inherently flip
    bot_values = np.flip(bot_values)
    bot_extended_values = np.concatenate([top_extended_values, bot_values], axis=0)

    return bot_extended_angles, bot_extended_values


def plot_ray_scene(angles, values):
    # flatten values and prepare for plotting
    angles_flattened = angles.reshape((-1, 2))
    thetas = angles_flattened[:, 0]
    phis = angles_flattened[:, 1]
    values_flattened = values.reshape((-1, 1))
    # plot details
    plt.scatter(x=phis, y=thetas, c=values_flattened, cmap="viridis", marker="o")
    plt.colorbar(label="Values")
    plt.axis("equal")
    plt.show()


def main() -> None:
    # CONSTANTS
    ITEM = "backpack"
    INF_DISTANCE = 3  # distance to use in case of no intersection
    ANGLE_THRESHOLD = math.pi / 2
    GRAB_THRESHOLD = 1.5
    ADD_BELOW = False
    PLOT_PROGRESSION = False
    RES = 128
    SIGMA = RES / 4
    RAY_START_COORDINATES = (-0.7, 0, 0.1)
    RAY_START_COORDINATES = None

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
    end_coordinates = np.mean(np.asarray(item_cloud.points), axis=0)
    if RAY_START_COORDINATES is not None:
        end_coordinates = np.asarray(RAY_START_COORDINATES)

    # get item and env meshes
    ball_sizes = (0.02, 0.011, 0.005)
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    item_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        item_cloud, radii=ball_sizes
    )
    environment_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        environment_cloud, radii=ball_sizes
    )
    item_mesh.paint_uniform_color(np.asarray([1, 0, 1]))

    # create a rayscene
    # convert to legacy
    # (from http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html)
    environment_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(environment_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(environment_mesh_legacy)

    # get rays
    ## get directions in both cartesian and spherical coordinates
    directions = get_uniform_sphere_directions(resolution=RES, return_cartesian=True)
    ## discard r slice of spherical coordinates (always == 1)
    sphere_angles = get_uniform_sphere_directions(
        resolution=RES, return_cartesian=False
    )[..., 1:]
    ## create rays (start_x, start_y, start_z, dir_x, dir_y, dir_z)
    ### first we set the starting points
    end_coordinates_tiled = np.tile(end_coordinates, (*directions.shape[:-1], 1))
    ### then we concatenate the directions
    rays = np.concatenate([end_coordinates_tiled, directions], axis=-1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ### cast rays and get intersection
    response = scene.cast_rays(rays)
    distances = response["t_hit"].numpy()
    distances[np.isinf(distances)] = INF_DISTANCE

    ####################################################################################
    ####################################################################################
    ################################### CASTING DONE ###################################
    ####################################################################################
    ################################## START PLOTTING ##################################
    ####################################################################################
    ####################################################################################

    angles = sphere_angles.copy()
    # initial plot
    plot_ray_scene(angles, distances)

    # first we create a boolean mask that gives us rays that either intersect after >=
    # GRAB_THRESHOLD, and the elevation of which is above ANGLE_THRESHOLD
    # these are the eligible angles
    angle_threshold = -math.pi + ANGLE_THRESHOLD
    valid_distances = distances > GRAB_THRESHOLD
    valid_angles = angles[..., 0] > angle_threshold
    valid_points = valid_distances & valid_angles
    if PLOT_PROGRESSION:
        plot_ray_scene(angles, valid_points)

    # here we extend the scene to take into account wrap-around (2pi = 0)
    angles, valid_points = repeat_scene(angles, valid_points, below=ADD_BELOW)

    # calculate the space I have to the left and the right for each ray
    distance_to_edge = steps_to_change(valid_points)
    # set elevations that are unblocked to be constant fully available
    full_available_mask = np.all(valid_points, axis=1)
    distance_to_edge[full_available_mask] = distance_to_edge.shape[1] // 2
    distance_to_edge[~valid_points] = 0
    if PLOT_PROGRESSION:
        plot_ray_scene(angles, distance_to_edge)

    # scale the distance according to the slice circumference at that elevation
    scale_angles = np.abs(angles[..., 0])
    scale = np.sin(scale_angles)
    # add a constant to the scale to not completely kill grabbing from above
    scale = scale + 0.5
    scaled_distance = scale * distance_to_edge
    scaled_distance[~valid_points] = -2
    if PLOT_PROGRESSION:
        plot_ray_scene(angles, scaled_distance)

    # add vertical gaussian blur
    gauss = create_quadratic_matrix(scaled_distance.shape[0], sigma=SIGMA)
    gaussian_scaled = gauss @ scaled_distance
    gaussian_scaled[~valid_points] = 0
    if PLOT_PROGRESSION:
        plot_ray_scene(angles, gaussian_scaled)

    # extract single image
    height, width = distances.shape[:2]
    original_angles = angles[height : height + height, width : width + width]
    original_values = gaussian_scaled[height : height + height, width : width + width]
    plot_ray_scene(original_angles, original_values)


if __name__ == "__main__":
    main()
