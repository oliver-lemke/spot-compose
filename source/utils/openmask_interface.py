import json
import os
import shutil
import zipfile

import numpy as np
import torch

import clip
import open3d as o3d
import requests
from urllib3.exceptions import ReadTimeoutError
from utils import recursive_config
from utils.docker_communication import _get_content
from utils.recursive_config import Config

MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device="cpu")


def zip_point_cloud(path: str) -> str:
    name = os.path.basename(path)
    if os.path.exists(name):
        shutil.rmtree(name)
    output_filename = os.path.join(path, f"{name}.zip")
    with zipfile.ZipFile(output_filename, "w") as zipf:
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".zip"):
                    continue
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, path))
    return output_filename


def get_mask_clip_features() -> None:
    # CONSTANTS
    PORT = 5001
    SAVE_PATH = "./tmp"

    config = recursive_config.Config()
    directory_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    directory_path = os.path.join(str(directory_path), ending)
    zipfile = zip_point_cloud(directory_path)

    kwargs = {
        "name": ("str", ending),
        "overwrite": ("bool", True),
        "scene_intrinsic_resolution": ("str", "[1440,1920]"),
        # "scene_intrinsic_resolution": ("str", "[968,1296]"),
    }
    server_address = f"http://localhost:{PORT}/openmask/save_and_predict"
    with open(zipfile, "rb") as f:
        try:
            response = requests.post(
                server_address, files={"scene": f}, params=kwargs, timeout=900
            )
        except ReadTimeoutError:
            print("Request timed out!")
            return

    if response.status_code == 200:  # fail
        contents = _get_content(response, SAVE_PATH)
    else:
        message = json.loads(response.content)
        print(f"{message['error']}", f"Status code: {response.status_code}", sep="\n")
        return

    features = contents["clip_features"]
    masks = contents["scene_MASKS"]

    save_path = config.get_subpath("openmask_features")
    save_path = os.path.join(save_path, ending)
    os.makedirs(save_path, exist_ok=False)
    feature_path = os.path.join(str(save_path), "clip_features.npy")
    mask_path = os.path.join(str(save_path), "scene_MASKS.npy")
    np.save(feature_path, features)
    np.save(mask_path, masks)

    # make unique
    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    features = features[mask_idx]
    feature_compressed_path = os.path.join(str(save_path), "clip_features_comp.npy")
    mask_compressed_path = os.path.join(str(save_path), "scene_MASKS_comp.npy")
    np.save(feature_compressed_path, features)
    np.save(mask_compressed_path, masks)


def get_mask_points(item: str, config, idx: int = 0, vis_block: bool = False):
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    base_path = config.get_subpath("openmask_features")
    feat_path = os.path.join(base_path, pcd_name, "clip_features_comp.npy")
    mask_path = os.path.join(base_path, pcd_name, "scene_MASKS_comp.npy")
    pcd_path = os.path.join(
        config.get_subpath("aligned_point_clouds"), pcd_name, "scene.ply"
    )

    features = np.load(feat_path)
    masks = np.load(mask_path)
    item = item.lower()

    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    # masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    # features = features[mask_idx]

    text = clip.tokenize([item]).to("cpu")

    # Compute the CLIP feature vector for the specified word
    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sim = torch.nn.functional.cosine_similarity(
        torch.Tensor(features), text_features, dim=1
    )
    values, indices = torch.topk(cos_sim, idx + 1)
    most_sim_feat_idx = indices[-1].item()
    print(f"{most_sim_feat_idx=}", f"value={values[-1].item()}")
    # idx = 1
    mask = masks[:, most_sim_feat_idx].astype(bool)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pcd_in = pcd.select_by_index(np.where(mask)[0])
    pcd_out = pcd.select_by_index(np.where(~mask)[0])

    if vis_block:
        pcd_in.paint_uniform_color([1, 0, 1])
        o3d.visualization.draw_geometries([pcd_in, pcd_out])

    return pcd_in, pcd_out


########################################################################################
# TESTING
########################################################################################


def visualize():
    item = "green watering can"
    config = Config()
    for i in range(5):
        print(i, end=", ")
        get_mask_points(item, config, idx=i, vis_block=True)


if __name__ == "__main__":
    # get_mask_clip_features()
    visualize()
