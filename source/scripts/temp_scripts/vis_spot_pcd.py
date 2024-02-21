import copy

import open3d as o3d
from utils.importer import TriangleMesh


def main() -> None:
    local_path = "/Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d/data/tmp/pcd.ply"
    # visualize
    pcd = o3d.io.read_point_cloud(local_path)
    sphere = TriangleMesh.create_sphere(radius=0.1)
    sphere2 = copy.deepcopy(sphere).translate([1, 0, 0])
    sphere.paint_uniform_color([0, 0, 0])
    sphere2.paint_uniform_color([0.5, 0.5, 0.5])

    print(pcd)
    o3d.visualization.draw_geometries([pcd, sphere, sphere2])


if __name__ == "__main__":
    main()
