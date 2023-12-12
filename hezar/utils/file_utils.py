from __future__ import annotations

import gzip
import json
import os
import shutil

import omegaconf

from .logging import Logger


logger = Logger(__name__)

__all__ = [
    "gunzip",
    "load_yaml_config",
    "load_json_config",
]


def load_yaml_config(path: str | os.PathLike):
    """
    Load yaml file using omegaconf
    """
    config = omegaconf.OmegaConf.load(path)
    return config


def load_json_config(path: str | os.PathLike):
    """
    Load json config file
    """
    with open(path) as f:
        config = json.load(f)
    return config


def gunzip(src_path, dest_path):
    """
    Unzip a .gz file from `src_path` and extract to `dest_path`
    Args:
        src_path: Path to .gz file
        dest_path: Path to the destination file

    Returns:

    """
    with gzip.open(src_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logger.debug(f"Extracted {src_path} to {dest_path}")
