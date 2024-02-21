"""
Utils for zero-shot object detection.
"""

from __future__ import annotations

import numpy as np

import cv2

# import skimage
from PIL import Image, ImageDraw
from transformers import pipeline
from utils.vis import normalize_image

_CHECKPOINT = "google/owlvit-base-patch32"
_DETECTOR = pipeline(model=_CHECKPOINT, task="zero-shot-object-detection")
_SCORE_THRESH = 0.2


def detect_objects(
    image: np.ndarray,
    items: list[str],
    input_format: str = "rgb",
    vis_block: bool = True,
) -> dict[str, dict]:
    """
    Detect objects in an image.
    :param image: to detect in
    :param items: which items to search for
    :param input_format: format of image, either "rgb" or "bgr"
    :param vis_block: whether to block execution for visualization
    :return: dict of detections consisting of {label: {score: score, box: (xmin, ymin, xmax, ymax)}}
    """
    input_format = input_format.lower()
    assert input_format in {"rgb", "bgr"}
    if input_format == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = normalize_image(image)

    image_pil = Image.fromarray(image)
    predictions = _DETECTOR(image_pil, candidate_labels=items)

    draw = ImageDraw.Draw(image_pil)
    return_dict = {}
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        return_dict[label] = {}
        return_dict[label]["score"] = score
        return_dict[label]["box"] = box
        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

    if vis_block:
        image_pil.show()

    return return_dict


def main() -> None:
    image = cv2.imread(
        "/Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d/source/scripts/my_robot_scripts/img.png"
    )
    # image = skimage.data.astronaut()
    objects = ["glasses", "cables"]
    dict_ = detect_objects(image, objects, input_format="rgb")

    print(f"{dict_=}")


if __name__ == "__main__":
    main()
