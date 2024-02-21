import json
import os
import shutil
import zipfile

import numpy as np

import requests
from urllib3.exceptions import ReadTimeoutError
from utils import recursive_config
from utils.docker_communication import _get_content


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


def main() -> None:
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

    per_mask_clip_features = contents["clip_features"]
    masks = contents["scene_MASKS"]

    save_path = config.get_subpath("openmask_features")
    save_path = os.path.join(save_path, ending)
    os.makedirs(save_path, exist_ok=False)
    feature_path = os.path.join(str(save_path), "clip_features.npy")
    mask_path = os.path.join(str(save_path), "scene_MASKS.npy")
    np.save(feature_path, per_mask_clip_features)
    np.save(mask_path, masks)


def get_mask_points(item: str, config):
    pass


if __name__ == "__main__":
    main()
