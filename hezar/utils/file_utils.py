import gzip
import shutil

from .logging import Logger


logger = Logger(__name__)

__all__ = [
    "gunzip"
]


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
