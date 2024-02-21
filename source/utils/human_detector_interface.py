"""
Util functions for human detector.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch

from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def detect_humans(image_array: np.ndarray, vis_block: bool = False) -> list[dict]:
    """
    Detects humans in the given image.

    :param image_array: The image data in numpy array format.
    :return: Detected humans with confidence scores and bounding box locations.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(np.flip(image_array, axis=-1))

    # Load the model and processor
    image_processor = AutoImageProcessor.from_pretrained(
        "devonho/detr-resnet-50_finetuned_cppe5"
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "devonho/detr-resnet-50_finetuned_cppe5"
    )

    detected_humans = []

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        # TODO: Why is this detecting as coverall?
        if model.config.id2label[label.item()] == "Coverall":
            detected_humans.append(
                {
                    "confidence": round(score.item(), 3),
                    "box": box,
                }
            )

        if vis_block:
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), model.config.id2label[label.item()], fill="white")

    if vis_block:
        image.show()
    return detected_humans


class BodyPart(Enum):
    """
    Enum for body parts
    """

    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"
    HEAD = "head"


def detect_body_part(
    depth_image: np.ndarray, body_part: BodyPart = BodyPart.RIGHT_HAND
) -> (float, np.ndarray):
    """
    Detect specific body parts.
    :param depth_image: depth image
    :param body_part: which body part to check for
    """
    raise NotImplementedError()
