# pylint: disable-all
import os
import sys

import numpy as np

import cv2
import open3d as o3d
from utils.coordinates import Pose3D
from utils.mask3D_interface import get_all_item_point_clouds
from utils.recursive_config import Config


def render_depth(geometries, camera):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    for geom in geometries:
        vis.add_geometry(geom)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera, True)

    # Capture depth buffer and create depth image
    depth = np.asarray(vis.capture_depth_float_buffer(True)) * 1000.0
    depth_scaled = depth.astype(np.uint16)
    image = np.asarray(vis.capture_screen_float_buffer(True))

    # Cleanup visualizer and return depth image
    vis.destroy_window()
    return depth_scaled, image


def main():
    # paths
    config = Config()
    data_path = config.get_subpath("data")
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    aligned_pcd_dir = os.path.join(data_path, "aligned_point_clouds", pcd_name)
    pre_pcd_dir = os.path.join(data_path, "prescans", pcd_name)
    pcd_path = os.path.join(aligned_pcd_dir, "scene.ply")
    icp_tform_ground_path = os.path.join(
        aligned_pcd_dir, "pose", "icp_tform_ground.txt"
    )
    intrinsics_path = os.path.join(aligned_pcd_dir, "intrinsic", "intrinsic_color.txt")
    mesh_path = os.path.join(pre_pcd_dir, "textured_output.obj")
    mask_name = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(data_path, "masked", mask_name)
    output_dir = os.path.join(data_path, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "img_rgb.png")
    segmentation_path = os.path.join(output_dir, "img_seg.png")

    # load
    mesh_ground = o3d.io.read_triangle_mesh(mesh_path, True)
    icp_tform_ground = np.loadtxt(icp_tform_ground_path)
    intrinsics = np.loadtxt(intrinsics_path)[:3, :3]
    print(intrinsics)

    # initial vis
    mesh = mesh_ground.transform(icp_tform_ground)
    colored_pcds = get_all_item_point_clouds(mask_path, pcd_path)
    # o3d.visualization.draw_geometries([mesh, pcd])

    # render image
    height, width = 1020, 1440
    cambase_tform_icp = Pose3D()
    cambase_tform_icp.set_rot_from_rpy((0, -90, 90), degrees=True)

    camera_tform_cambase = Pose3D((3, 0, 1))
    camera_tform_cambase.set_rot_from_direction((-1, -1, -0.3), roll=-5, degrees=True)
    camera_tform_icp = cambase_tform_icp @ camera_tform_cambase.inverse()

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, intrinsic_matrix=intrinsics
    )
    camera.extrinsic = camera_tform_icp.as_matrix()
    depth, image = render_depth([mesh, *colored_pcds], camera)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image Title", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(img_path, image_rgb * 255)

    depth, image = render_depth([mesh, *colored_pcds], camera)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image Title", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(segmentation_path, image_rgb * 255)
    # cv2.imwrite(depth_path, depth)


if __name__ == "__main__":
    main()
    sys.exit(0)
