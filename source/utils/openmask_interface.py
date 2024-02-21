import copy
import os
import shutil
import sys
import zipfile

import numpy as np
import torch

import clip
import open3d as o3d

# from transformers import CLIPModel, CLIPProcessor
from utils import recursive_config
from utils.docker_communication import send_request
from utils.files import prep_tmp_path
from utils.importer import PointCloud


def _zip_point_cloud(path: str) -> str:
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


def predict_masks_and_features(
    data_directory: str, config: recursive_config.Config
) -> None:
    # CONSTANTS
    TIMEOUT = 180

    zip_path = _zip_point_cloud(data_directory)
    name = os.path.basename(data_directory)

    address_details = config["servers"]["openmask"]
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    kwargs = {"name": (str, name), "overwrite": [bool, True]}
    paths_dict = {"scene": zip_path}
    tmp_path = prep_tmp_path(config)

    contents = send_request(address, paths_dict, kwargs, TIMEOUT, tmp_path)

    per_mask_clip_features = contents["clip_features"]
    masks = np.argmax(contents["scene_MASKS"], axis=1)

    save_path = config.get_subpath("openmask_features")
    save_path = os.path.join(save_path, name)
    os.makedirs(save_path, exist_ok=False)
    feature_path = os.path.join(str(save_path), "clip_features.npy")
    mask_path = os.path.join(str(save_path), "scene_MASKS.npy")
    np.save(feature_path, per_mask_clip_features)
    np.save(mask_path, masks)


def get_mask_points(
    item: str, config: recursive_config.Config, vis_block: bool = True
) -> PointCloud:
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    # image = np.zeros((1, 3, 224, 224))
    #
    # inputs = processor(text=[item], images=image, return_tensors="pt", padding=True)
    #
    # outputs = model(**inputs)
    # text_embed = outputs.text_embeds.detach().numpy()
    # print(f"{text_embed=}")

    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", torch.device("cuda"))
    # clip_model, clip_preprocess = clip.load("ViT-L/14@336px", torch.device("cuda"))
    # clip_model, clip_preprocess = clip.load("ViT-L/14", torch.device("cuda"))
    text_input_processed = clip.tokenize(item).to(torch.device("cuda"))
    with torch.no_grad():
        text_embed = clip_model.encode_text(text_input_processed)
    text_embed = text_embed.detach().cpu().numpy()

    text_embed_norm = np.linalg.norm(text_embed)

    mask_base_path = config.get_subpath("openmask_features")
    name = config["pre_scanned_graphs"]["high_res"]
    mask_feature_path = os.path.join(str(mask_base_path), name, "clip_features.npy")
    mask_features = np.load(mask_feature_path)
    mask_features_norm = np.linalg.norm(mask_features, axis=1)

    cosine_similarity = text_embed @ mask_features.T
    score = cosine_similarity / (text_embed_norm * mask_features_norm + 1e-15)
    score[cosine_similarity == 0] = -1
    score = score.squeeze()
    closest = np.argmax(score)
    print(f"Highest Score: mask {closest} with {score[closest]}.")
    #
    mask_path = os.path.join(str(mask_base_path), name, "scene_MASKS.npy")
    masks = np.load(mask_path)
    print(list(np.where(np.max(mask_features, axis=1) > 0)))
    idx = np.where(masks[..., closest])

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")
    pc = o3d.io.read_point_cloud(pc_path)

    item_pc = pc.select_by_index(idx[0])
    env_pc = pc.select_by_index(idx[0], invert=True)

    if vis_block:
        item_pc_vis = copy.deepcopy(item_pc).paint_uniform_color((1, 0, 0))
        env_pc_vis = copy.deepcopy(env_pc).paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([item_pc_vis])
        o3d.visualization.draw_geometries([item_pc_vis + env_pc])
    return item_pc


def main() -> None:
    config = recursive_config.Config()
    item = "rug"
    get_mask_points(item, config, vis_block=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
