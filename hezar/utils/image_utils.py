from __future__ import annotations

from io import BytesIO
from typing import Iterable, Tuple

import numpy as np
import requests
import torch

from ..constants import Backends, ChannelsAxisSide, ImageType
from .common_utils import is_url
from .integration_utils import is_backend_available
from .logging import Logger


logger = Logger(__name__)

if is_backend_available(Backends.PILLOW):
    from PIL import Image

__all__ = [
    "convert_image_type",
    "normalize_image",
    "load_image",
    "show_image",
    "rescale_image",
    "resize_image",
    "mirror_image",
    "gray_scale_image",
    "find_channels_axis_side",
    "transpose_channels_axis_side",
]


def verify_image_dims(image: np.ndarray):
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Image input must be a numpy array of size 2 or 3! Got {image.shape}")


def convert_image_type(
    image: np.ndarray | "Image" | torch.Tensor,
    target_type: str | ImageType = ImageType.NUMPY,
):
    """
    Convert image lib type. Supports numpy array, pillow image and torch tensor.
    """
    if isinstance(image, Image.Image):
        if image.mode == "L":
            image = np.asarray(image)
            image = np.expand_dims(image, 0)
        else:
            image = np.asarray(image)
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    verify_image_dims(image)

    if target_type == ImageType.PILLOW:
        # transpose channels to the last axis since pillow cannot handle it otherwise
        if find_channels_axis_side(image) == ChannelsAxisSide.FIRST:
            image = transpose_channels_axis_side(
                image,
                axis_side=ChannelsAxisSide.LAST,
                src_axis_side=ChannelsAxisSide.FIRST,
            )
        num_channels = image.shape[0] if find_channels_axis_side(image) == ChannelsAxisSide.FIRST else image.shape[-1]
        if num_channels == 1:
            image = image[:, :, -1]
            image = Image.fromarray(image, "L")
        else:
            image = Image.fromarray(image)
    elif target_type == ImageType.TORCH:
        image = torch.tensor(image)

    return image


def load_image(path, return_type: str | ImageType = ImageType.PILLOW):
    """
    Load an image file to a desired return format

    Args:
        path: Path to image file
        return_type: Image output type ("pillow", "numpy", "torch")

    Returns:
        The desired output image of type `PIL.Image` or `numpy.ndarray` or `torch.Tensor`
    """
    if is_url(path):
        pil_image = Image.open(BytesIO(requests.get(path).content)).convert("RGB")
    else:
        pil_image = Image.open(path).convert("RGB")
    converted_image = convert_image_type(pil_image, return_type)
    return converted_image


def show_image(image: "Image" | torch.Tensor | np.ndarray, title: str = "Image"):
    """
    Given any type of input image (PIL, numpy, torch), show the image in a window

    Args:
        image: Input image of types PIL.Image, numpy.ndarray or torch.Tensor
        title: Optional title for the preview window
    """
    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image.show(title=title)


def rescale_image(image: np.ndarray, scale: float):
    verify_image_dims(image)
    image = image * scale
    return image


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    resample=None,
    reducing_gap: float = None,
    return_type: ImageType = ImageType.NUMPY,
):
    """
    Resize a numpy array image (actually uses pillow PIL.Image.resize(...))

    Args:
        image: Numpy image
        size: A tuple of (width, height)
        resample: Resampling filter (refer to PIL.Image.Resampling) for possible values
        reducing_gap: Optimization method for resizing based on reducing times
        return_type: Return type of the image (numpy, torch, pillow)

    Returns:
        The resized image
    """
    verify_image_dims(image)
    if len(size) != 2:
        raise ValueError(f"The value of `size` must be a 2-sized tuple! Got length {len(size)}(`{size}`)")
    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image = pil_image.resize(size, resample=resample, reducing_gap=reducing_gap)
    np_image = convert_image_type(pil_image, return_type)
    return np_image


def mirror_image(image: np.ndarray, return_type: str | ImageType = ImageType.NUMPY):
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    verify_image_dims(image)

    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    final_image = convert_image_type(pil_image, return_type)
    return final_image


def gray_scale_image(image: np.ndarray, return_type: str | ImageType = ImageType.NUMPY):
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    verify_image_dims(image)

    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image = pil_image.convert("L")
    np_image = convert_image_type(pil_image, ImageType.NUMPY)
    final_image = convert_image_type(np_image, target_type=return_type)
    return final_image


def normalize_image(
    image: np.ndarray,
    mean: float | Iterable[float],
    std:  float | Iterable[float],
    channel_axis: str | ChannelsAxisSide = "first",
):
    verify_image_dims(image)

    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    num_channels = image.shape[0 if channel_axis == ChannelsAxisSide.FIRST else -1]

    if not isinstance(mean, Iterable):
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if not isinstance(std, Iterable):
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if channel_axis == ChannelsAxisSide.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    return image


def find_channels_axis_side(image: np.ndarray, num_channels: int = None):
    valid_num_channels = (num_channels,) if num_channels is not None else (1, 2, 3)
    if image.shape[0] in valid_num_channels:
        return ChannelsAxisSide.FIRST
    else:
        return ChannelsAxisSide.LAST


def transpose_channels_axis_side(
    image: np.ndarray,
    axis_side:  str | ChannelsAxisSide,
    num_channels: int = None,
    src_axis_side: str | ChannelsAxisSide = None,
):
    """
    Convert an image channels axis side from (channels, ...) to (..., channels) or vise versa.

    Args:
        image: Input image
        axis_side: The desired axis side (can be "first" or "last")
        num_channels: The number of channels in the input image
        src_axis_side: The image initial channels axis side (can be "first" or "last")

    Returns:
        The image with the converted channels axis
    """
    if src_axis_side is None:
        src_axis_side = find_channels_axis_side(image, num_channels=num_channels)

    # If input's channels axis side and output channels axis side are the same return the same image
    if src_axis_side == axis_side:
        return image

    if axis_side == ChannelsAxisSide.FIRST:
        image = image.transpose((2, 0, 1))
    elif axis_side == ChannelsAxisSide.LAST:
        image = image.transpose((1, 2, 0))

    return image
