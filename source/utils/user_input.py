"""
Util functions for getting user input
"""

from __future__ import annotations

import numpy as np

from utils.coordinates import Pose2D
from utils.docker_interfaces.mask3D_interface import is_valid_label


def get_wanted_item_mask3d(max_retries: int = 3) -> str:
    """
    Ask user for the item he wants to find in Mask3D segmented input.
    :param max_retries: how often to re-ask if given invalid input
    """
    for _ in range(max_retries):
        answer = input("Please type in the item you want to pick up: ").strip()
        if is_valid_label(answer):
            return answer
        else:
            print(f"Item {answer} is not a valid label!")
    raise ValueError("Invalid input limit exceeded!")


def _pprint_ndarray(array: np.ndarray) -> str:
    """
    Pretty-print a numpy array.
    """
    lst = array.reshape((-1,)).tolist()
    return str(lst)


def get_yes_no_answer(prompt: str, max_retries: int = 3, default: bool = False) -> bool:
    """
    Get a yes or no answer to a question. Style [prompt] (y/n)?
    :param prompt: question to ask, do not include question mark (?)
    :param max_retries: how many times to re-ask given invalid input
    :param default: default answer if given repeated invalid input
    """
    for _ in range(max_retries):
        answer = input(f"{prompt} (y/n)? ").strip()
        if answer in {"y", "n"}:
            return answer == "y"
        else:
            print("Do not understand answer, please answer 'y' (yes) or 'n' (no).")
    return default


def confirm_coordinates(
    start_pose: Pose2D,
    end_pose: Pose2D,
    destination_pose: Pose2D,
    distance: float,
) -> bool:
    """
    Confirm movement to distanced position.
    :param start_pose: starting coordinates
    :param end_pose: theoretical target pose
    :param destination_pose: actual target pose (distanced by distance from end_pose)
    :param distance: distance to stop away from end_pose
    """
    print(f"The robot is currently at starting pose {start_pose=}.")
    print(f"The calculated end pose is {end_pose=}.")
    print(f"With {distance=}, the walking destination is {destination_pose=}.")
    answer = get_yes_no_answer("Do you want to confirm this action")
    assert answer
    return answer


def confirm_move(
    start_pose: Pose2D,
    end_pose: Pose2D,
):
    """
    Confirm movement to a position.
    :param start_pose: starting coordinates
    :param end_pose: target pose
    """
    print(f"The robot is currently at starting pose {start_pose=}.")
    print(f"The calculated end pose is {end_pose=}.")
    answer = get_yes_no_answer("Do you want to confirm this action")
    assert answer
    return answer


def get_n_word_answer(
    prompt: str, nr_words: int = -1, max_retries: int = 3, add_linebreak: bool = True
) -> list[str]:
    """
    Get an answer consisting of n words.
    :param prompt: Question to ask, include question mark (?)
    :param nr_words: number of words to ask for, -1 meaning no limit
    :param max_retries: how many times to re-ask given invalid input
    :param add_linebreak: whether to add a linebreak at the end of the prompt
    """
    if add_linebreak:
        prompt += "\n"
    for _ in range(max_retries):
        response = input(prompt)
        words = response.strip().split()
        if not (nr_words in (-1, len(words))):
            print(f"Need at least {nr_words} words, these are {len(words)}")
        else:
            return words
    raise ValueError("Could not get object")
