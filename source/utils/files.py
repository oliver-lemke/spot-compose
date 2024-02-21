"""
Util functions for file management.
"""

import os
import shutil

from utils.recursive_config import Config


def prep_tmp_path(config: Config) -> str:
    """
    Creates a temporary folder at specified directory in config.
    Deletes all prior files at said path beforehand.
    """
    tmp_path = config.get_subpath("tmp")
    if os.path.exists(tmp_path):
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        elif os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=False)
    return tmp_path
