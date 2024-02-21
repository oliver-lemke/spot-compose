"""
Util functions for segmenting point clouds with Mask3D.
"""

from __future__ import annotations

import os.path

import numpy as np

import open3d as o3d
import pandas as pd
from utils import recursive_config
from utils.importer import PointCloud
from utils.scannet_200_labels import CLASS_LABELS_200, VALID_CLASS_IDS_200


def is_valid_label(item: str) -> bool:
    """
    Check whether a label is valid (within the specified possible labels).
    """
    return item in CLASS_LABELS_200


def _get_list_of_items(folder_path: str) -> pd.DataFrame:
    """
    Read the csv that includes the output from Mask3D.
    It is a 3-column csv, which specifies the file that indexes the points belonging to the object, its label, and the
    confidence.
    :param folder_path: path of folder containing the csv
    :return: pandas dataframe of the csv
    """
    dir_, base = os.path.split(folder_path)
    csv_path = os.path.join(dir_, base, f"{base}.txt")
    df = pd.read_csv(csv_path, delimiter=" ", header=None)
    df.columns = ["path_ending", "class_label", "confidence"]
    return df


def get_coordinates_from_item(
    item: str,
    folder_path: str | bytes,
    point_cloud_path: str | bytes,
    index: int = 0,
) -> (PointCloud, PointCloud):
    """
    Given an item description, we extract all points that are part of this item.
    Returns two point clouds, one representing the item, the other the rest.
    :param item: name of item to extract
    :param folder_path: base folder path of the Mask3D output
    :param point_cloud_path: path for the point cloud
    :param index: if there are multiple objects for a given label, which one to focus on
    """
    if not is_valid_label(item):
        raise ValueError(f"Item {item} is not a valid label")
    # convert string label to numeric
    idx = CLASS_LABELS_200.index(item)
    label = VALID_CLASS_IDS_200[idx]

    df = _get_list_of_items(str(folder_path))
    # get all entries for our item label
    entries = df[df["class_label"] == label]
    if index > len(entries) or index < (-len(entries) + 1):
        index = 0
    # get "index" numbered object
    entry = entries.iloc[index]
    path_ending = entry["path_ending"]

    # get the mask of the item
    mask_file_path = os.path.join(folder_path, path_ending)
    with open(mask_file_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    good_points_bool = np.asarray([bool(int(line)) for line in lines])

    # read the point cloud, select by indices specified in the file
    pc = o3d.io.read_point_cloud(point_cloud_path)

    good_points_idx = np.where(good_points_bool)[0]
    environment_cloud = pc.select_by_index(good_points_idx, invert=True)
    item_cloud = pc.select_by_index(good_points_idx)

    return item_cloud, environment_cloud


def test() -> None:
    config = recursive_config.Config()

    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), f"{ending}.ply")

    res = get_coordinates_from_item("backpack", mask_path, pc_path)
    print(res)


if __name__ == "__main__":
    test()
