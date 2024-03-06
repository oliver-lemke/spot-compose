from __future__ import annotations

import os.path
from logging import Logger
from typing import Optional

import numpy as np

import cv2
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils.docker_communication import save_files, send_request
from utils.files import prep_tmp_path
from utils.object_detetion import BBox, Detection, Match
from utils.recursive_config import Config
from utils.vis import draw_boxes

COLORS = {
    "door": (0.651, 0.243, 0.957),
    "handle": (0.522, 0.596, 0.561),
    "cabinet door": (0.549, 0.047, 0.169),
    "refrigerator door": (0.082, 0.475, 0.627),
}

CATEGORIES = {"0": "door", "1": "handle", "2": "cabinet door", "3": "refrigerator door"}


def predict_yolodrawer(
    image: np.ndarray,
    config: Config,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    input_format: str = "rgb",
    vis_block: bool = False,
) -> list[Detection] | None:
    assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    address_details = config["servers"]["yolodrawer"]
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    tmp_path = prep_tmp_path(config)

    save_data = [("image.npy", np.save, image)]
    image_path, *_ = save_files(save_data, tmp_path)

    paths_dict = {"image": image_path}
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, {}, timeout, tmp_path)
    if logger:
        logger.info("Received response!")

    # no detections
    if len(contents) == 0:
        if vis_block:
            draw_boxes(image, [])
        return []

    classes = contents["classes"]
    confidences = contents["confidences"]
    bboxes = contents["bboxes"]

    detections = []
    for cls, conf, bbox in zip(classes, confidences, bboxes):
        name = CATEGORIES[str(int(cls))]
        det = Detection(name, conf, BBox(*bbox))
        detections.append(det)

    if vis_block:
        draw_boxes(image, detections)

    return detections


def predict_darknet(
    image: np.ndarray,
    config: Config,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    input_format: str = "rgb",
    vis_block: bool = False,
) -> list[Detection] | None:
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
        logger.info("Received response!")

    # no detections
    if len(contents) == 0:
        if vis_block:
            draw_boxes(image, [])
        return []

    detections = contents["detections"]
    # detections are in format (x_center, y_center, width, height)
    if vis_block:
        draw_boxes(image, detections)

    def convert_format(detection: list[str, float, list[float]]) -> Detection:
        (x, y, w, h) = detection[2]
        xmin, xmax = x - w / 2, x + w / 2
        ymin, ymax = y - h / 2, y + h / 2
        return Detection(detection[0], detection[1], BBox(xmin, ymin, xmax, ymax))

    detections = [convert_format(det) for det in detections]
    return detections


# noinspection PyTypeChecker
def drawer_handle_matches(detections: list[Detection]) -> list[Match]:
    def calculate_ioa(drawer: Detection, handle: Detection) -> float:
        _, _, drawer_bbox = drawer
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

        intersection_area = overlap_width * overlap_height
        handle_area = (handle_right - handle_left) * (handle_bottom - handle_top)

        ioa = intersection_area / handle_area
        return ioa

    def matching_score(
        drawer: Detection, handle: Detection, ioa_weight: float = 10.0
    ) -> tuple[float, float]:
        _, drawer_conf, _ = drawer
        ioa = calculate_ioa(drawer, handle)
        if ioa == 0:
            return ioa, ioa
        else:
            score = ioa_weight * ioa + drawer_conf
            return score, ioa

    drawer_detections = [det for det in detections if det.name == "cabinet door"]
    handle_detections = [det for det in detections if det.name == "handle"]

    matching_scores = np.zeros((len(drawer_detections), len(handle_detections), 2))
    for didx, drawer_detection in enumerate(drawer_detections):
        for hidx, handle_detection in enumerate(handle_detections):
            matching_scores[didx, hidx] = np.array(
                matching_score(drawer_detection, handle_detection)
            )
    drawer_idxs, handle_idxs = linear_sum_assignment(-matching_scores[..., 0])
    matches = [
        Match(drawer_detections[drawer_idx], handle_detections[handle_idx])
        for (drawer_idx, handle_idx) in zip(drawer_idxs, handle_idxs)
        if matching_scores[drawer_idx, handle_idx, 1] > 0.9  # ioa
    ]

    for drawer_idx, drawer_detection in enumerate(drawer_detections):
        if drawer_idx not in drawer_idxs:
            matches.append(Match(drawer_detection, None))

    for handle_idx, handle_detection in enumerate(handle_detections):
        if handle_idx not in handle_idxs:
            matches.append(Match(None, handle_detection))

    return matches


########################################################################################
########################################################################################
####################################### TESTING ########################################
########################################################################################
########################################################################################


def _test_pose() -> None:
    config = Config()
    base_path = config.get_subpath("data")
    dir_path = os.path.join(base_path, "test", "iCloud Photos")
    image_names = [
        "IMG_1885.jpg",
        "IMG_1886.jpg",
        "IMG_1887.jpg",
        "IMG_1888.jpg",
        "IMG_1889.jpg",
        "IMG_1890.jpg",
    ]
    for image_name in image_names:
        img_path = os.path.join(dir_path, image_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _ = predict_yolodrawer(image, config, vis_block=True)


if __name__ == "__main__":
    _test_pose()
