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
from utils.docker_interfaces.docker_communication import _get_content
from utils.recursive_config import Config
from utils.importer import PointCloud

MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device="cpu")
_MIN_MASK_CONFIDENCE_DEFAULT = 0.3
_MIN_CLIP_SIMILARITY_DEFAULT = 0.23


def zip_point_cloud(path: str) -> str:
    name = os.path.basename(path)
    if os.path.exists(name):
        shutil.rmtree(name)
    output_filename = os.path.join(path, f"{name}.zip")
    with zipfile.ZipFile(output_filename, "w") as zipf:
        for foldername, _, filenames in os.walk(path):
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
    zip_file = zip_point_cloud(directory_path)

    num_queries = 120
    kwargs = {
        "name": ("str", ending),
        "overwrite": ("bool", True),
        "scene_intrinsic_resolution": ("str", "[1440,1920]"),
        # "scene_intrinsic_resolution": ("str", "[968,1296]"),
        "num_queries": ("int", num_queries),
    }
    server_address = f"http://localhost:{PORT}/openmask/save_and_predict"
    with open(zip_file, "rb") as f:
        try:
            response = requests.post(
                server_address, files={"scene": f}, params=kwargs, timeout=6000
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
    masks = masks.T
    scores = contents["scene_SCORES"]
    classes = contents["scene_CLASSES"]

    save_path = config.get_subpath("openmask_features")
    save_path = os.path.join(save_path, ending)
    os.makedirs(save_path, exist_ok=False)
    feature_path = os.path.join(str(save_path), "clip_features.npy")
    mask_path = os.path.join(str(save_path), "scene_MASKS.npy")
    score_path = os.path.join(str(save_path), "scene_SCORES.npy")
    class_path = os.path.join(str(save_path), "scene_CLASSES.npy")
    np.save(feature_path, features)
    np.save(mask_path, masks)
    np.save(score_path, scores)
    np.save(class_path, classes)

    # make unique
    masks, mask_idx = np.unique(masks, axis=0, return_index=True)
    scores = scores[mask_idx]
    classes = classes[mask_idx]
    features = features[mask_idx]
    feature_compressed_path = os.path.join(str(save_path), "clip_features_comp.npy")
    mask_compressed_path = os.path.join(str(save_path), "scene_MASKS_comp.npy")
    score_compressed_path = os.path.join(str(save_path), "scene_SCORES_comp.npy")
    class_compressed_path = os.path.join(str(save_path), "scene_CLASSES_comp.npy")
    np.save(feature_compressed_path, features)
    np.save(mask_compressed_path, masks)
    np.save(score_compressed_path, scores)
    np.save(class_compressed_path, classes)


def get_item_masks(
    item: str,
    config: Config,
    min_mask_confidence: float = _MIN_MASK_CONFIDENCE_DEFAULT,
    min_clip_similarity: float = _MIN_CLIP_SIMILARITY_DEFAULT,
    comp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the masks of the relevant items

    Args:
        item (str): the item to be searched for
        config (Config): the configuration
        min_mask_confidence (float, optional): minimum confidence in the Mask3D mask to be considered (filters low likelihood objects). Defaults to _MIN_MASK_CONFIDENCE_DEFAULT.
        min_clip_similarity (float, optional): minimum similarity score between 3D object and item description to be considered. Defaults to _MIN_CLIP_SIMILARITY_DEFAULT.
        comp (bool, optional): whether to use the unique masks. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (cos_sims, masks, object_ids, mask_scores)
        (1) cos_sims: the cosine similarity between the object and the item description (sorted)
        (2) masks: the masks of the 3D objects
        (3) object_ids: the ids of the objects
        (4) mask_scores: Mask3D confidence score in that mask
    """
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    base_path = config.get_subpath("openmask_features")
    if comp:
        feat_path = os.path.join(base_path, pcd_name, "clip_features_comp.npy")
        mask_path = os.path.join(base_path, pcd_name, "scene_MASKS_comp.npy")
        score_path = os.path.join(base_path, pcd_name, "scene_SCORES_comp.npy")
    else:
        feat_path = os.path.join(base_path, pcd_name, "clip_features.npy")
        mask_path = os.path.join(base_path, pcd_name, "scene_MASKS.npy")
        score_path = os.path.join(base_path, pcd_name, "scene_SCORES.npy")

    features = np.load(feat_path)
    masks = np.load(mask_path)
    mask_scores = np.load(score_path)
    object_ids = np.arange(features.shape[0])
    item = item.lower()

    pass_score_bool = mask_scores > min_mask_confidence
    masks = masks[pass_score_bool]
    mask_scores = mask_scores[pass_score_bool]
    features = features[pass_score_bool]
    object_ids = object_ids[pass_score_bool]

    text = clip.tokenize([item]).to("cpu")

    # Compute the CLIP feature vector for the specified word
    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sims = torch.cosine_similarity(torch.Tensor(features), text_features, dim=1)
    detections_above_min_sim_score = cos_sims > min_clip_similarity
    masks = masks[detections_above_min_sim_score]
    cos_sims = cos_sims[detections_above_min_sim_score]
    object_ids = object_ids[detections_above_min_sim_score]

    return cos_sims, masks, object_ids, mask_scores


def get_item_pcd(
    item: str,
    config,
    idx: int = 0,
    vis_block: bool = False,
    comp: bool = True,
    min_mask_confidence: float = _MIN_MASK_CONFIDENCE_DEFAULT,
    min_clip_similarity: float = _MIN_CLIP_SIMILARITY_DEFAULT,
) -> tuple[PointCloud, PointCloud]:
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    pcd_path = os.path.join(
        config.get_subpath("aligned_point_clouds"), pcd_name, "scene.ply"
    )

    cos_sim, masks, _, scores = get_item_masks(
        item, config, min_mask_confidence, min_clip_similarity, comp
    )
    nr_detections = cos_sim.shape[0]
    if idx + 1 > nr_detections:
        print(f"Not enough detections for {nr_detections=} and {idx=}!")
        return None, None

    values, indices = torch.topk(cos_sim, idx + 1)
    most_sim_feat_idx = indices[-1].item()
    print(f"{most_sim_feat_idx=}", f"value={values[-1].item()}", end=", ")
    print(f"mask_confidence={scores[most_sim_feat_idx].item()}")
    mask = masks[most_sim_feat_idx].astype(bool)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    print(mask.shape, pcd)
    pcd_in = pcd.select_by_index(np.where(mask)[0])
    pcd_out = pcd.select_by_index(np.where(~mask)[0])

    points_in = np.asarray(pcd_in.points)
    bbox_min = np.min(points_in, axis=0)
    bbox_max = np.max(points_in, axis=0)
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)

    if vis_block:
        pcd_in.paint_uniform_color([1, 0, 1])
        o3d.visualization.draw_geometries([pcd_in, pcd_out, aabb])

    return pcd_in, pcd_out


def compute_bounding_boxes(
    config,
    vis_block: bool = False,
    min_mask_confidence: bool = _MIN_MASK_CONFIDENCE_DEFAULT,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    base_path = config.get_subpath("openmask_features")
    feat_path = os.path.join(base_path, pcd_name, "clip_features_comp.npy")
    mask_path = os.path.join(base_path, pcd_name, "scene_MASKS_comp.npy")
    score_path = os.path.join(base_path, pcd_name, "scene_SCORES_comp.npy")
    class_path = os.path.join(base_path, pcd_name, "scene_CLASSES_comp.npy")
    pcd_path = os.path.join(
        config.get_subpath("aligned_point_clouds"), pcd_name, "scene.ply"
    )

    features = np.load(feat_path)
    masks = np.load(mask_path)
    scores = np.load(score_path)
    # classes = np.load(class_path)
    object_ids = np.arange(features.shape[0])

    pass_score_bool = scores > min_mask_confidence
    masks = masks[pass_score_bool]
    scores = scores[pass_score_bool]
    features = features[pass_score_bool]
    object_ids = object_ids[pass_score_bool]

    masks = masks.astype(bool)
    nr_objects = masks.shape[0]
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)

    # get the mins and maxes per object
    maskss = np.repeat(masks[..., np.newaxis], 3, axis=2)
    pointss = np.repeat(points[np.newaxis, ...], nr_objects, axis=0)
    masked_pointss = np.ma.array(
        pointss, mask=~maskss
    )  # True means masked, opposite meaning therefore ~
    bbox_mins = masked_pointss.min(axis=1)
    bbox_maxs = masked_pointss.max(axis=1)
    nr_bboxes = bbox_maxs.shape[0]

    if vis_block:
        aabbs = []
        for i in range(nr_bboxes):
            bbox_min = bbox_mins[i]
            bbox_max = bbox_maxs[i]

            # Create an axis-aligned bounding box
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=bbox_min, max_bound=bbox_max
            )
            aabbs.append(aabb)

        # Optionally, visualize the bounding box
        o3d.visualization.draw_geometries([pcd, *aabbs])

    return (bbox_mins, bbox_maxs), features, object_ids


def get_scene_dict(
    config,
    vis_block: bool = False,
    min_mask_confidence: bool = _MIN_MASK_CONFIDENCE_DEFAULT,
) -> dict:
    (bbox_mins, bbox_maxs), _, object_ids = compute_bounding_boxes(
        config, vis_block, min_mask_confidence
    )
    nr_bboxes = bbox_mins.shape[0]
    objects_dict = {}
    for idx in range(nr_bboxes):
        object_id = object_ids[idx]
        bbox_min, bbox_max = bbox_mins[idx], bbox_maxs[idx]
        centroid = (bbox_min + bbox_max) / 2
        extents = (bbox_max - bbox_min) / 2
        current_object_dict = {
            "description": "irrelevant",
            "centroid": centroid.tolist(),
            "extents": extents.tolist(),
        }
        objects_dict[f"object_{object_id}"] = current_object_dict
    scene_dict = {"scene": objects_dict}
    return scene_dict


########################################################################################
# TESTING
########################################################################################


def visualize_query_masks():
    item = "candle"
    config = Config()
    for i in range(5):
        print(i, end=", ")
        success = get_item_pcd(
            item,
            config,
            idx=i,
            vis_block=True,
            min_mask_confidence=0.0,
            min_clip_similarity=0.1,
        )
        if not success:
            break


def visualize_bounding_boxes():
    os.environ["DISPLAY"] = "localhost:11.0"
    config = Config()
    compute_bounding_boxes(config, vis_block=True, min_mask_confidence=0.3)


if __name__ == "__main__":
    # get_mask_clip_features()
    visualize_query_masks()
    # visualize_bounding_boxes()
