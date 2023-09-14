from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

from ..builders import build_preprocessor
from ..configs import PreprocessorConfig
from ..constants import DEFAULT_IMAGE_PROCESSOR_CONFIG_FILE, DEFAULT_PREPROCESSOR_SUBFOLDER, ImageType, Backends
from ..registry import register_preprocessor
from ..utils import (
    convert_batch_dict_dtype,
    convert_image_type,
    load_image,
    normalize_image,
    rescale_image,
    resize_image,
    transpose_channels_axis_side,
)
from .preprocessor import Preprocessor

_required_backends = [
    Backends.PILLOW,
]

_DESCRIPTION = r"""
A general image processor to do all the image transforms in a composable/configurable pipeline
"""

_image_type_aliases = {
    "pt": ImageType.TORCH,
    "pytorch": ImageType.TORCH,
    "torch": ImageType.TORCH,
    "np": ImageType.NUMPY,
    "numpy": ImageType.NUMPY,
    "pil": ImageType.PIL,
    "pillow": ImageType.PIL,
}


@dataclass
class ImageProcessorConfig(PreprocessorConfig):
    name = "image_processor"
    mean: List[float] = None
    std: List[float] = None
    rescale: float = None
    resample: int = None
    size: Tuple[int, int] = field(
        default=None,
        metadata={"description": "Image size tuple (width, height)"},
    )


@register_preprocessor("image_processor", config_class=ImageProcessorConfig, description=_DESCRIPTION)
class ImageProcessor(Preprocessor):
    """
    General image processor to perform sequential transforms on the images
    """
    required_backends = _required_backends

    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER
    image_processor_config_file = DEFAULT_IMAGE_PROCESSOR_CONFIG_FILE

    def __init__(self, config: ImageProcessorConfig, **kwargs):
        super().__init__(config, **kwargs)

    def __call__(
        self,
        images: List,
        mean: float = None,
        std: float = None,
        rescale: float = None,
        size: Tuple[int, int] = None,
        resample: float = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """
        Perform a sequential image processing on a list of input images. You can control the behavior directly from
        this method's parameters.

        Args:
            images: A list of input images of types torch, numpy, pillow.
            mean: Image mean value for normalization
            std: Image std value for normalization
            rescale: Scale factor for rescaling the image
            size: A tuple of (width, height) to resize the images
            resample: Resample method value based on Image.Resampling
            return_tensors: The type of the output images
            **kwargs: Extra parameters

        Returns:
            The dict of transformed images list
        """
        mean = mean or self.config.mean
        std = std or self.config.std
        rescale = rescale or self.config.rescale
        size = size or self.config.size
        resample = resample or self.config.resample

        # Load images if inputs are list of files
        images = [load_image(x, return_type="numpy") if isinstance(x, str) else x for x in images]

        # Cast image types
        images = [convert_image_type(image, target_type="numpy") for image in images]

        if size is not None:
            if not isinstance(size, Iterable) or len(size) > 2:
                raise ValueError(f"The parameter `size` must be a tuple/list of (width, height), got `{size}`")
            images = [resize_image(image, size=size, resample=resample) for image in images]

        if rescale is not None:
            images = [rescale_image(image, scale=rescale) for image in images]

        if mean is not None and std is not None:
            images = [normalize_image(image, mean=mean, std=std, channel_axis="last") for image in images]

        # transpose channels axis
        images = [transpose_channels_axis_side(image, axis_side="first") for image in images]

        # Return images batch dict
        images = np.array([convert_image_type(image, target_type="numpy") for image in images], dtype=np.float32)

        images = convert_batch_dict_dtype({"pixel_values": images}, dtype=return_tensors)

        return images

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder: str = None,
        force_return_dict: bool = False,
        config_filename: str = None,
        **kwargs,
    ) -> "ImageProcessor":
        subfolder = subfolder or cls.preprocessor_subfolder
        config_filename = config_filename or cls.image_processor_config_file
        config = ImageProcessorConfig.load(hub_or_local_path, filename=config_filename, subfolder=subfolder)
        image_processor = build_preprocessor(config.name, config, **kwargs)
        return image_processor

    def save(
        self,
        path,
        subfolder=None,
        config_filename=None,
    ):
        subfolder = subfolder or self.preprocessor_subfolder
        config_filename = config_filename or self.image_processor_config_file

        self.config.save(path, subfolder=subfolder, filename=config_filename)

    def push_to_hub(
        self,
        repo_id,
        subfolder=None,
        commit_message=None,
        private=None,
        config_filename=None
    ):
        subfolder = subfolder or self.preprocessor_subfolder
        config_filename = config_filename or self.image_processor_config_file

        if commit_message is None:
            commit_message = "Hezar: Upload image processor files"

        self.config.push_to_hub(
            repo_id,
            subfolder=subfolder,
            filename=config_filename,
            private=private,
            commit_message=commit_message,
        )
