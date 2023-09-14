from typing import Iterable, Tuple, Union

import numpy as np
import torch

from ..constants import Backends, ChannelsAxisSide, ImageType
from .integration_utils import is_backend_available
from .logging import Logger


logger = Logger(__name__)

if is_backend_available(Backends.PILLOW):
    from PIL import Image

__all__ = [
    "convert_image_type",
    "normalize_image",
    "load_image",
    "rescale_image",
    "resize_image",
    "find_channels_axis_side",
    "transpose_channels_axis_side",
]


def verify_image_dims(image: np.ndarray):
    if len(image.shape) != 3:
        raise ValueError(f"Image input must be a numpy array of size 3! Got {image.shape}")


def convert_image_type(
    image: Union[np.ndarray, "Image", torch.Tensor],
    target_type: Union[str, ImageType] = ImageType.NUMPY,
):
    """
    Convert image lib type. Supports numpy array, pillow image and torch tensor.
    """
    if isinstance(image, Image.Image) and target_type != ImageType.PIL:
        image = np.asarray(image)
    elif isinstance(image, torch.Tensor) and target_type != ImageType.TORCH:
        image = image.numpy()

    verify_image_dims(image)

    if target_type == ImageType.PIL:
        image = Image.fromarray(image)
    elif target_type == ImageType.TORCH:
        image = torch.tensor(image)

    return image


def load_image(path, return_type: Union[str, ImageType] = ImageType.PIL):
    pil_image = Image.open(path).convert("RGB")
    converted_image = convert_image_type(pil_image, return_type)
    return converted_image


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
        return_type: Return type of the image (numpy, torch, pil)

    Returns:
        The resized image
    """
    verify_image_dims(image)
    if len(size) != 2:
        raise ValueError(f"The value of `size` must be a 2-sized tuple! Got length {len(size)}(`{size}`)")
    pil_image = convert_image_type(image, ImageType.PIL)
    pil_image = pil_image.resize(size, resample=resample, reducing_gap=reducing_gap)
    np_image = convert_image_type(pil_image, return_type)
    return np_image


def normalize_image(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    channel_axis: Union[str, ChannelsAxisSide] = "first",
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


def find_channels_axis_side(image: np.ndarray, num_channels: int = 3):
    if image.shape[0] == num_channels:
        return ChannelsAxisSide.FIRST
    else:
        return ChannelsAxisSide.LAST


def transpose_channels_axis_side(
    image: np.ndarray,
    axis_side: Union[str, ChannelsAxisSide],
    num_channels: int = 3,
    src_axis_side: Union[str, ChannelsAxisSide] = None,
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
