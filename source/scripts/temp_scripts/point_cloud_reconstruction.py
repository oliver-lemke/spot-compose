import numpy as np

import open3d as o3d
from utils.coordinates import (
    Pose2D,
    Pose3D,
    get_uniform_sphere_directions,
    pose_distanced,
)
from utils.importer import PointCloud


def main() -> None:
    # CONSTANTS
    DISTANCE = 0.2
    DIM = DISTANCE / 2
    RES = 4

    # create base_cloud seen by every arm
    ## create mesh of sphere
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=DIM / 2)
    # mesh = o3d.geometry.TriangleMesh.create_box(width=DIM, height=DIM, depth=DIM)
    # mesh.translate([-DIM / 2] * 3)
    base_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)
    points = base_cloud.points

    ## translate the point cloud in the x-direction
    translation_vector = np.array([0.2, 0.0, 0.0])
    points += translation_vector

    ## create a new point cloud with the translated points
    base_cloud = o3d.geometry.PointCloud()
    base_cloud.points = o3d.utility.Vector3dVector(points)

    directions = get_uniform_sphere_directions(resolution=RES, return_cartesian=True)
    rot_matrices = []
    for direction in directions:
        direction = Pose3D(direction)
        position = pose_distanced(direction, distance=DISTANCE)

        matrix = np.zeros((4, 4))
        matrix[:3, :3] = position.rot.to_matrix()
        matrix[:3, 3] = position.get_as_ndarray()
        matrix[3, 3] = 1

        rot_matrices.append(matrix)

    rectified_clouds = []
    for idx, rot_matrix in enumerate(rot_matrices):
        copy_cloud = PointCloud(base_cloud.points)
        t_matrix = rot_matrix
        rectified_cloud = copy_cloud.transform(t_matrix)

        nr_clouds = len(rot_matrices)
        color = np.asarray([0, 0, ((1 + idx) / nr_clouds)])
        rectified_cloud.paint_uniform_color(color)

        rectified_clouds.append(rectified_cloud)

    mesh.paint_uniform_color(np.asarray([0, 1, 0]))
    clouds = [*rectified_clouds, mesh]
    o3d.visualization.draw_geometries(clouds)


if __name__ == "__main__":
    main()
