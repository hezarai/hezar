from __future__ import annotations

import os
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
    from PIL import Image, ImageDraw

__all__ = [
    "convert_image_type",
    "normalize_image",
    "load_image",
    "show_image",
    "save_image",
    "rescale_image",
    "resize_image",
    "mirror_image",
    "gray_scale_image",
    "find_channels_axis_side",
    "transpose_channels_axis_side",
    "draw_boxes",
    "crop_boxes",
    "pad_boxes",
]


def verify_image_dims(image: np.ndarray):
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Image input must be a numpy array of size 2 or 3! Got {image.shape}")


def convert_image_type(
    image: np.ndarray | Image.Image | torch.Tensor,
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


def show_image(image: Image.Image | torch.Tensor | np.ndarray, title: str = "Image"):
    """
    Given any type of input image (PIL, numpy, torch), show the image in a window

    Args:
        image: Input image of types PIL.Image, numpy.ndarray or torch.Tensor
        title: Optional title for the preview window
    """
    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image.show(title=title)


def save_image(image: Image.Image | torch.Tensor | np.ndarray, path: str):
    """
    Save a PIL/numpy/torch image into a path.

    Args:
        image: A PIL.Image or np.ndarray or torch.Tensor image type
        path: The full path of the image to save
    """
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    pil_image = convert_image_type(image, ImageType.PILLOW)
    pil_image.save(path)


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
    std: float | Iterable[float],
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
    axis_side: str | ChannelsAxisSide,
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


def draw_boxes(image, bboxes, bbox_color: tuple = (0, 255, 0)) -> "Image.Image":
    """
    Draw bbox on the image

    Args:
        image: A *single* image (Pillow Image object)
        bboxes: A list of bboxes in a single image
        bbox_color: Color of the bbox in image (RGB tuple)

    Returns:
        The overlaid image
    """
    # Convert the bbox_color from BGR to RGB if necessary
    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        if bbox is None:
            continue
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        # Draw the rectangle on the image
        draw.rectangle((x1, y1, x2, y2), outline=bbox_color, width=1)

    return image


def pad_boxes(bboxes, padding: int | tuple = None):
    """
    Add a padding to sides of the bounding boxes.

    Args:
        bboxes: A list of bounding boxes (x1, y1, w, h)
        padding: A single integer value to pad equally or a tuple of size 4 for more specific padding. Tuple order is
            (left, up, right, down)

    Returns:
        A list of padded bounding boxes
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    if isinstance(padding, tuple) and len(padding) != 4:
        raise ValueError(f"padding must be a single int value or a tuple of size 4, got {len(padding)}!")

    padded_bboxes = []
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        x1 -= padding[0]
        x2 += padding[2]
        y1 -= padding[1]
        y2 += padding[3]

        w = x2 - x1
        h = y2 - y1
        padded_bboxes.append((x1, y1, w, h))
    return padded_bboxes


def crop_boxes(image, bboxes, padding: int | tuple = None) -> list["Image.Image"]:
    """
    Crop all bounding boxes in an image

    Args:
        image: A *single* image (Pillow Image object)
        bboxes: A list of bboxes in a single image (x1, y1, width, height)
        padding: Number of pixels to pad the bbox coordinates. An int value means all sides and tuple pads sides
            specifically.

    Returns:
        A list of cropped images
    """
    if padding is not None:
        bboxes = pad_boxes(bboxes, padding)

    cropped_images = []
    for bbox in bboxes:
        x1, y1, w, h = bbox

        cropped_image = image.crop((x1, y1, x1 + w, y1 + h))
        cropped_images.append(cropped_image)

    return cropped_images
