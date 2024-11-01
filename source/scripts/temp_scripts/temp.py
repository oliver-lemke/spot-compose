import open3d as o3d
import numpy as np

# Load the point cloud
point_cloud = o3d.io.read_point_cloud(
    "/home/olemke/workspace/code/spot-compose-v2/resources/aligned_point_clouds/24-10-29/scene.ply"
)
point_cloud = o3d.io.read_point_cloud(
    "/home/olemke/workspace/code/spot-compose-v2/resources/prescans/24-10-29/pcd.ply"
)

point_cloud = point_cloud.translate((1, 1, 1))
point_cloud = point_cloud.rotate(np.eye(3))

o3d.visualization.draw_geometries([point_cloud])

print(point_cloud)
