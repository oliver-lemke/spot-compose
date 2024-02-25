from __future__ import annotations

import os.path
from logging import Logger
from typing import Optional

import numpy as np

import cv2
from matplotlib import pyplot as plt

from utils.docker_communication import save_files, send_request
from utils.files import prep_tmp_path
from utils.recursive_config import Config

COLORS = {"door": (0.651, 0.243, 0.957),
          "handle": (0.522, 0.596, 0.561),
          "cabinet door": (0.549, 0.047, 0.169),
          "refrigerator door": (0.082, 0.475, 0.627)}


def draw_boxes(image: np.ndarray,
               detections: list[tuple[str, float, list[float]]]) -> None:
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    for name, conf, (x, y, w, h) in detections:
        xmin, ymin = x - w / 2, y - h / 2
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                   fill=False, color=COLORS[name], linewidth=6))
        text = f'{name}: {conf:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def predict(
        image: np.ndarray,
        config: Config,
        logger: Optional[Logger] = None,
        timeout: int = 90,
        input_format: str = "rgb",
        vis_block: bool = False,
):
    assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    address_details = config["servers"]["darknet"]
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    tmp_path = prep_tmp_path(config)

    save_data = [("image.npy", np.save, image)]
    image_path, *_ = save_files(save_data, tmp_path)

    paths_dict = {"image": image_path}
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, {}, timeout, tmp_path)
    if logger:
        logger.info(f"Received response!")

    if len(contents) == 0:
        return None

    detections = contents["detections"]
    # detections are in format (x_center, y_center, width, height)

    if vis_block:
        draw_boxes(image, detections)

    return detections


########################################################################################
########################################################################################
####################################### TESTING ########################################
########################################################################################
########################################################################################


def _test_pose() -> None:
    config = Config()
    base_path = config.get_subpath("data")
    image_name = "img.jpg"
    # image_name = "landscape.png"
    img_path = os.path.join(base_path, image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = predict(image, config, vis_block=True)
    print(keypoints)


if __name__ == "__main__":
    _test_pose()
