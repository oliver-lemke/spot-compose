from __future__ import annotations

import os.path
from logging import Logger
from typing import Optional

import numpy as np

import cv2
from utils.docker_interfaces.docker_communication import save_files, send_request
from utils.files import prep_tmp_path
from utils.recursive_config import Config


def predict(
    image: np.ndarray,
    config: Config,
    top_n: int = 1,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    input_format: str = "rgb",
    vis_block: bool = False,
):
    assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kwargs = {
        "top_n": ("int", top_n),
        "vis": ("bool", vis_block),
    }

    address_details = config["servers"]["vitpose"]
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    tmp_path = prep_tmp_path(config)

    save_data = [("image.npy", np.save, image)]
    image_path, *_ = save_files(save_data, tmp_path)

    paths_dict = {"image": image_path}
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, kwargs, timeout, tmp_path)
    if logger:
        logger.info("Received response!")

    if len(contents) == 0:
        return None

    # get gripper meshes (already transformed)
    keypoints = contents["keypoints"]

    if vis_block:
        detections_image = contents["detections_image"]
        cv2.imshow("Keypoint Detections", detections_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints


########################################################################################
########################################################################################
####################################### TESTING ########################################
########################################################################################
########################################################################################


def _test_pose() -> None:
    config = Config()
    base_path = config.get_subpath("resources")
    image_name = "human.jpg"
    # image_name = "landscape.png"
    img_path = os.path.join(base_path, "test", image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = predict(image, config, top_n=1, vis_block=True)
    print(keypoints)


if __name__ == "__main__":
    _test_pose()
