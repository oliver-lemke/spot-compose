from __future__ import annotations

import os.path
from collections import namedtuple
from logging import Logger
from typing import Optional

import numpy as np

import cv2
from matplotlib import pyplot as plt
from utils.docker_communication import save_files, send_request
from utils.files import prep_tmp_path
from utils.recursive_config import Config

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["name", "conf", "bbox"])

COLORS = {
    "door": (0.651, 0.243, 0.957),
    "handle": (0.522, 0.596, 0.561),
    "cabinet door": (0.549, 0.047, 0.169),
    "refrigerator door": (0.082, 0.475, 0.627),
}


def draw_boxes(image: np.ndarray, detections: list[Detection]) -> None:
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    for name, conf, (x, y, w, h) in detections:
        xmin, ymin = x - w / 2, y - h / 2
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), w, h, fill=False, color=COLORS[name], linewidth=6
            )
        )
        text = f"{name}: {conf:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def predict(
    image: np.ndarray,
    config: Config,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    input_format: str = "rgb",
    vis_block: bool = False,
) -> list[Detection]:
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

    detections = [Detection(det[0], det[1], BBox(*det[2])) for det in detections]
    return detections


def drawer_handle_tuples(detections: list[Detection]) -> list[tuple[BBox | None]]:
    def matching_cost(drawer: Detection, handle: Detection) -> float:
        _, drawer_conf, drawer_bbox = drawer
        *_, handle_bbox = handle

        # calculate overlap
        handle_left, handle_top, handle_right, handle_bottom = handle_bbox
        drawer_left, drawer_top, drawer_right, drawer_bottom = drawer_bbox

        # Calculate the overlap between the bounding boxes
        overlap_left = max(handle_left, drawer_left)
        overlap_top = max(handle_top, drawer_top)
        overlap_right = min(handle_right, drawer_right)
        overlap_bottom = min(handle_bottom, drawer_bottom)

        # Calculate the area of the overlap
        overlap_width = max(0, overlap_right - overlap_left)
        overlap_height = max(0, overlap_bottom - overlap_top)

        overlap_area = overlap_width * overlap_height
        handle_area = (handle_right - handle_left) * (handle_bottom - handle_top)
        return overlap_area / handle_area

    drawer_detections = [det for det in detections if det.name == "cabinet door"]
    handle_detections = [det for det in detections if det.name == "handle"]

    matching_costs = np.zeros((len(drawer_detections), len(handle_detections)))
    for didx, drawer_detection in enumerate(drawer_detections):
        for hidx, handle_detection in enumerate(handle_detections):
            matching_costs[didx, hidx] = matching_cost(
                drawer_detection, handle_detection
            )
    print(matching_costs)


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
