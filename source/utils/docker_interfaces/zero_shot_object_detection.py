"""
Utils for zero-shot object detection.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import cv2
import requests
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from utils import vis
from utils.docker_interfaces.object_detection import BBox, Detection
from utils.vis import normalize_image

_CHECKPOINT = "google/owlv2-base-patch16-ensemble"
_PROCESSOR = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
_MODEL = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
_SCORE_THRESH = 0.5


def detect_objects(
    image: np.ndarray,
    items: list[str],
    input_format: str = "rgb",
    add_photo_of: bool = True,
    vis_block: bool = True,
) -> list[Detection]:
    """
    Detect objects in an image.
    :param image: to detect in
    :param items: which items to search for
    :param input_format: format of image, either "rgb" or "bgr"
    :param add_photo_of: whether to add 'a photo of' to the input
    :param vis_block: whether to block execution for visualization
    :return: dict of detections consisting of {label: {score: score, box: (xmin, ymin, xmax, ymax)}}
    """
    input_format = input_format.lower()
    assert input_format in {"rgb", "bgr"}
    if input_format == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = normalize_image(image)
    if add_photo_of:
        texts = [f"a photo of a {item}" for item in items]
    else:
        texts = items

    image_pil = Image.fromarray(image)
    inputs = _PROCESSOR(text=[texts], images=image_pil, return_tensors="pt")
    outputs = _MODEL(**inputs)
    target_sizes = torch.Tensor([image_pil.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = _PROCESSOR.post_process_object_detection(
        outputs=outputs, threshold=_SCORE_THRESH, target_sizes=target_sizes
    )
    predictions = results[0]

    detections = []
    scores = predictions["scores"].cpu().detach().numpy()
    labels = predictions["labels"].cpu().detach().numpy()
    boxes = predictions["boxes"].cpu().detach().numpy()
    for box, score, label in zip(boxes, scores, labels):
        bbox = BBox(*box)
        detection = Detection(name=items[label], conf=score, bbox=bbox)
        detections.append(detection)

    if vis_block:
        vis.draw_boxes(image, detections)

    return detections


def main() -> None:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True, timeout=10).raw)
    texts = ["cat", "dog"]
    start = time.time_ns()
    detections = detect_objects(np.asarray(image), texts, vis_block=True)
    end = time.time_ns()
    print(end - start)

    print(f"{detections=}")


if __name__ == "__main__":
    main()
