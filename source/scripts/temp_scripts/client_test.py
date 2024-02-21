# pylint: disable-all
from __future__ import annotations

import os.path

import numpy as np

import open3d as o3d
from utils import recursive_config
from utils.coordinates import (
    Coordinates,
    get_uniform_sphere_directions,
    grasp_from_direction,
    remove_duplicate_rows,
)
from utils.docker_communication import save_files, send_request
from utils.importer import PointCloud
from utils.mask3D_interface import get_coordinates_from_item


def get_armadillo_model(
    scale: float = 1.0, vis: bool = False, return_mesh: bool = False
) -> PointCloud | (PointCloud, PointCloud):
    """
    Create model of armadillo for testing
    :param scale: scale of model
    :param vis: visualize model before returning
    :param return_mesh: return mesh in addition to point cloud
    :return: point cloud and optionally mesh
    """
    armadillo_mesh = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.asarray([0.5] * 3))
    if scale != 1.0:
        mesh.scale(scale, np.zeros((3, 1)))

    point_cloud = mesh.sample_points_uniformly(5000)

    if vis:
        o3d.visualization.draw_geometries([point_cloud])

    if return_mesh:
        return point_cloud, mesh
    else:
        return point_cloud


def get_point_cloud(
    item: str,
    scale: float,
    radius: float,
    index: int = 0,
    vis: bool = False,
):
    """
    Extract item point cloud from given full point cloud
    :param item: item to extract
    :param scale: scale item afterward to adjust to grip
    :param radius: radius around item to include
    :param index: index for getting item
    :param vis: visualize model before returning
    :return: point cloud and optionally mesh
    """
    # get the aligned point cloud
    config = recursive_config.Config()
    directory_path = config.get_subpath("aligned_point_clouds")
    directory_path = os.path.join(
        str(directory_path), config["pre_scanned_graphs"]["high_res"]
    )
    pc_path = os.path.join(directory_path, "scene.ply")

    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    # get the point cloud associated with the item
    item_cloud, env_cloud = get_coordinates_from_item(
        item, mask_path, pc_path, index=index
    )
    points = np.asarray(item_cloud.points)
    center = points.mean(axis=0)
    mins, maxs = np.min(points, axis=0), np.max(points, axis=0)
    limits = np.stack([mins, maxs], axis=0)

    env_tree = o3d.geometry.KDTreeFlann(env_cloud)
    _, idx, _ = env_tree.search_radius_vector_3d(center, radius)
    radius_cloud = env_cloud.select_by_index(idx)

    # scale point cloud (bc. max gripper width is 0.1)
    cloud = item_cloud + radius_cloud

    # create mesh from item point cloud
    # need to scale ball sizes in accordance with point cloud
    ball_sizes = np.asarray((0.02, 0.011, 0.005))
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, radii=ball_sizes
    )
    if vis:
        o3d.visualization.draw_geometries([mesh])

    cloud = cloud.scale(scale, np.zeros((3, 1)))
    item_cloud = item_cloud.scale(scale, np.zeros((3, 1)))
    mesh = mesh.scale(scale, np.zeros((3, 1)))
    limits = limits * scale

    return cloud, item_cloud, mesh, limits


def get_gripper_mesh(scale: float = 1.0, color: (float, float, float) = (1, 0, 1)):
    """
    Get mesh of gripper for visualization
    :param scale: of gripper
    :param color: of gripper
    :return: mesh of gripper
    """
    mesh = o3d.io.read_triangle_mesh("./gripper.ply")
    mesh = mesh.scale(scale, np.zeros((3, 1)))
    mesh.paint_uniform_color(color)
    return mesh


def get_rotation_matrices(resolution: int) -> np.ndarray:
    """
    The network only predicts grasps from the current perspective. To get more
    possibilities, we have to look at the object from a lot of different angles.
    To do this, we specify a bunch of rotation matrices to rotate the object.
    :param resolution: how many angles angular degree
    :return: rotation matrices of shape (nr_angles, 3, 3)
    """
    # first get some uniform directions
    directions = get_uniform_sphere_directions(resolution)
    directions = directions.reshape((-1, 3))
    # in this case we don't want duplicate rows
    directions = remove_duplicate_rows(directions, tolerance=1e-5)
    rot_matrices = []
    invariant_direction = Coordinates((0, 0, 1))
    # convert from directions to rotation matrices to look at the object from there
    for direction in directions:
        direction = Coordinates(direction)
        coords = grasp_from_direction(
            direction,
            invariant_direction=invariant_direction,
            distance=0,
            roll=0,
            use_degrees=True,
        )
        rot_matrix = coords.rot.to_matrix()
        rot_matrices.append(rot_matrix)
    rot_matrices = np.stack(rot_matrices)
    return rot_matrices


def _filter(contents, item_cloud, limits, thresh: float = 0.02) -> list[(int, int)]:
    scoress = contents["scoress"]
    tf_matricess = contents["tf_matricess"]

    points = np.asarray(item_cloud.points)

    mins = limits[0]
    maxs = limits[1]
    center = np.asarray([0, 0, 0, 1])
    indices = []
    for idx_rot, tf_matrices in enumerate(tf_matricess):
        for idx_nr, tf_matrix in enumerate(tf_matrices):
            if scoress[idx_rot, idx_nr] == -1:
                continue
            grip_point = tf_matrix @ center
            grip_point = grip_point[:3] / grip_point[3]
            outside = np.any((grip_point < mins) | (grip_point > maxs))
            if outside:
                continue
            grip_point = np.expand_dims(grip_point, 0)
            distances = np.linalg.norm(points - grip_point, axis=1, ord=2)
            min_distance = np.min(distances, axis=0)
            if min_distance >= thresh:
                continue
            indices.append((idx_rot, idx_nr))
    return indices


def main() -> None:
    # CONSTANTS
    PORT = 5000  # where to find docker
    TOP_N = 1  # how many positions from each angle
    VIS = True  # visualize grippers after
    SAVE_PATH = "./tmp/"  # where to save files for correspondence
    ITEM, INDEX = "lamp", 3  # which item point cloud to extract
    ITEM, INDEX = "bottle", 0  # which item point cloud to extract
    ROTATION_RESOLUTION = 8  # how many angles
    RADIUS = 0.3

    # max gripper width is 0.175m, but in model is 0.100m, therefore we scale models
    scale = 0.1 / 0.175
    # inv_scale = 1 / scale

    # get point cloud
    # point_cloud, mesh = get_armadillo_model(scale=0.001, vis=False, return_mesh=True)
    point_cloud, item_cloud, mesh, limits = get_point_cloud(
        ITEM, scale=scale, radius=RADIUS, index=INDEX, vis=False
    )
    # The network only predicts grasps from the current perspective. To get more
    # possibilities, we have to look at the object from a lot of different angles.
    # To do this, we specify a bunch of rotation matrices to rotate the object.
    # get the rotation matrices for the different viewing angles
    rotations = get_rotation_matrices(ROTATION_RESOLUTION)
    # rotations = np.eye(3).reshape((1, 3, 3))
    save_data = [
        ("points.npy", np.save, np.asarray(point_cloud.points)),
        ("colors.npy", np.save, np.asarray(point_cloud.colors)),
        ("limits.npy", np.save, limits),
        ("rotations.npy", np.save, rotations),
    ]
    points_path, colors_path, limits_path, rotations_path = save_files(
        save_data, SAVE_PATH
    )

    # prepare request to gsnet in docker (these are function arguments)
    kwargs = {
        "gripper_height": ["float", 0.05],
        "top_n": ["int", TOP_N],
        "vis": ["bool", VIS],
    }
    server_address = f"http://localhost:{PORT}/graspnet/predict"
    # create server request by attaching all the files in "files" and the arguments in
    paths_dict = {
        "points": points_path,
        "colors": colors_path,
        "limits": limits_path,
        "rotations": rotations_path,
    }
    contents = send_request(server_address, paths_dict, kwargs, 20, SAVE_PATH)

    # get gripper meshes (already transformed)
    scoress = contents["scoress"]
    argmax = np.unravel_index(np.argmax(scoress), scoress.shape)
    good_indices = _filter(contents, item_cloud, limits)
    grippers = {
        (rot, nr): contents[f"mesh_{rot:03}_{nr:03}"] for rot, nr in good_indices
    }
    # visualize all best per angle
    o3d.visualization.draw_geometries([mesh, *grippers.values()])
    # visualize best gripper only
    o3d.visualization.draw_geometries([mesh, grippers[argmax]])
    print("Done")


if __name__ == "__main__":
    main()
