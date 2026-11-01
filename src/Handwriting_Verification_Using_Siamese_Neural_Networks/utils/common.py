import os
from pathlib import Path

import yaml
from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations

from Handwriting_Verification_Using_Siamese_Neural_Networks.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox object.
    """

    try:
        with open(path_to_yaml) as yaml_file:

            content = yaml.safe_load(yaml_file)

            logger.info(f"yaml file: {path_to_yaml} loaded successfully")

            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")

    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list):
    """
    Creates directories if they do not exist.
    """

    for path in path_to_directories:

        os.makedirs(path, exist_ok=True)

        logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns file size in KB.
    """

    size_in_kb = round(os.path.getsize(path) / 1024)

    return f"~ {size_in_kb} KB"