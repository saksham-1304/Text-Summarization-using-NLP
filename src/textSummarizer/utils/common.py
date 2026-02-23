"""Common utility functions used across the pipeline.

Provides YAML reading, directory creation, file size checking,
and JSON save/load helpers.
"""

import os
import json
from typing import Any, List
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

from textSummarizer.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return as ConfigBox for dot-access.

    Args:
        path_to_yaml: Path to the YAML file.

    Returns:
        ConfigBox wrapping the YAML content.

    Raises:
        ValueError: If the YAML file is empty.
        FileNotFoundError: If the file does not exist.
    """
    path_to_yaml = Path(path_to_yaml)
    if not path_to_yaml.exists():
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            logger.info(f"YAML file loaded: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")


def create_directories(path_to_directories: List[str], verbose: bool = True) -> None:
    """Create a list of directories if they don't exist.

    Args:
        path_to_directories: List of directory paths to create.
        verbose: If True, log each directory creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    """Get file size in human-readable format.

    Args:
        path: Path to the file.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    size_bytes = os.path.getsize(path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def save_json(path: Path, data: dict) -> None:
    """Save a dictionary as JSON file.

    Args:
        path: Destination file path.
        data: Dictionary to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"JSON saved: {path}")


def load_json(path: Path) -> dict:
    """Load a JSON file as dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary with the JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON loaded: {path}")
    return content

    
