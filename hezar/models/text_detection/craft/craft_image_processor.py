from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from ....preprocessors.image_processor import ImageProcessor, ImageProcessorConfig
from ....registry import register_preprocessor
from ....utils import (
    convert_batch_dict_dtype,
    convert_image_type,
    load_image,
    mirror_image,
    normalize_image,
    resize_image,
    transpose_channels_axis_side,
)


@dataclass
class CraftImageProcessorConfig(ImageProcessorConfig):
    name = "craft_image_processor"
    mean: tuple[float] = (123.675, 116.28, 103.53)
    std: tuple[float] = (58.395, 57.12, 57.375)
    rescale: float = None
    square_size: int = 2560
    mag_ratio: float = 1.0


@register_preprocessor("craft_image_processor", config_class=CraftImageProcessorConfig)
class CraftImageProcessor(ImageProcessor):
    def __init__(self, config: CraftImageProcessorConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def get_ratio(self, image, square_size: int = None, mag_ratio: float = None):
        square_size = square_size or self.config.square_size
        mag_ratio = mag_ratio or self.config.mag_ratio

        height, width, channel = image.shape

        target_size = mag_ratio * max(height, width)
        target_size = min(target_size, square_size)

        ratio = target_size / max(height, width)

        return ratio

    def _resize(self, image, square_size: int, mag_ratio: float):
        """
        Resizes the image by pasting it into a bigger canvas
        """
        height, width, channel = image.shape

        # magnify image size
        target_size = mag_ratio * max(height, width)

        # set original image size
        target_size = min(target_size, square_size)

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        inner_image = resize_image(image, size=(target_w, target_h))

        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized_image = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized_image[0:target_h, 0:target_w, :] = inner_image

        return resized_image

    def __call__(
        self,
        images: List,
        device: str = None,
        mean: float = None,
        std: float = None,
        rescale: float = None,
        square_size: tuple[int] = None,
        mag_ratio: float = None,
        mirror: bool = None,
        gray_scale: bool = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        mean = mean or self.config.mean
        std = std or self.config.std
        square_size = square_size or self.config.square_size
        mag_ratio = mag_ratio or self.config.mag_ratio
        mirror = mirror or self.config.mirror

        if not isinstance(images, Iterable) or isinstance(images, str):
            images = [images]

        # Load images if inputs are list of files
        images = [load_image(x, return_type="numpy") if isinstance(x, str) else x for x in images]

        # Cast image types
        images = [convert_image_type(image, target_type="numpy") for image in images]

        # Mirror images if mirror is set
        if mirror:
            images = [mirror_image(image, return_type="numpy") for image in images]

        ratio_values = [self.get_ratio(image) for image in images]
        images = [self._resize(image, square_size=square_size, mag_ratio=mag_ratio) for image in images]

        if mean is not None and std is not None:
            images = [normalize_image(image, mean=mean, std=std, channel_axis="last") for image in images]

        # Transpose channels axis
        images = [transpose_channels_axis_side(image, axis_side="first") for image in images]

        # Return images batch dict
        images = np.array([convert_image_type(image, target_type="numpy") for image in images], dtype=np.float32)

        outputs = convert_batch_dict_dtype({"pixel_values": images}, dtype=return_tensors)

        if device:
            import torch

            outputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}

        outputs["ratio_values"] = ratio_values

        return outputs
