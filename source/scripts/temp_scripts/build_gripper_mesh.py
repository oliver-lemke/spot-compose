import copy

import numpy as np

import open3d as o3d
from scipy.spatial.transform import Rotation


def get_gripper_mesh(scale: float = 1.0, color=(1, 0, 1)):
    # Create base boxes
    box1 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 1)
    rot = np.eye(4)
    rot[:2, 3] = np.asarray([-0.05] * 2)
    box1.transform(rot)

    box3 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 2)
    rot = np.eye(4)
    rot[:2, 3] = np.asarray([-0.05] * 2)
    box3.transform(rot)

    # Create copies of the boxes
    box2 = copy.deepcopy(box1)
    box4 = copy.deepcopy(box3)
    box5 = copy.deepcopy(box3)

    # base of the gripper
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
    box1.transform(rot)
    box2.transform(np.linalg.inv(rot))

    # fingers of the gripper
    box3.translate([0, -1, 0])
    box4.translate([0, 1, 0])

    # arm of the gripper
    box5.translate([0, 0, -2])

    # Combine the boxes
    combined = box1 + box2 + box3 + box4 + box5
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("xyz", [0, 90, 0], degrees=True).as_matrix()
    rot[:3, 3] = np.asarray([-1, 0, 0])
    combined.transform(rot)

    # center sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    combined = combined + sphere
    combined.paint_uniform_color(np.asarray(color))

    return combined.scale(scale, np.asarray([0, 0, 0]))


def main():
    mesh = get_gripper_mesh(scale=1)
    o3d.visualization.draw_geometries([mesh])
    if True:
        o3d.io.write_triangle_mesh("./gripper.ply", mesh)


if __name__ == "__main__":
    main()
