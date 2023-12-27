from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

from ..builders import build_preprocessor
from ..configs import PreprocessorConfig
from ..constants import (
    DEFAULT_IMAGE_PROCESSOR_CONFIG_FILE,
    DEFAULT_PREPROCESSOR_SUBFOLDER,
    Backends,
    ImageType,
)
from ..registry import register_preprocessor
from ..utils import (
    convert_batch_dict_dtype,
    convert_image_type,
    gray_scale_image,
    load_image,
    mirror_image,
    normalize_image,
    rescale_image,
    resize_image,
    transpose_channels_axis_side,
)
from .preprocessor import Preprocessor


# List of backends required for the image processor
_required_backends = [
    Backends.PILLOW,
]

_DESCRIPTION = r"""
A general image processor to perform various image transformations in a composable/configurable pipeline.
"""

# Aliases for different image types
_image_type_aliases = {
    "pt": ImageType.TORCH,
    "pytorch": ImageType.TORCH,
    "torch": ImageType.TORCH,
    "np": ImageType.NUMPY,
    "numpy": ImageType.NUMPY,
    "pil": ImageType.PILLOW,
    "pillow": ImageType.PILLOW,
}


@dataclass
class ImageProcessorConfig(PreprocessorConfig):
    """
    Configuration class for the ImageProcessor.
    """
    name = "image_processor"
    mean: List[float] = None
    std: List[float] = None
    rescale: float = None
    resample: int = None
    size: Tuple[int, int] = field(
        default=None,
        metadata={"description": "Image size tuple (width, height)"},
    )
    mirror: bool = False
    gray_scale: bool = False


@register_preprocessor("image_processor", config_class=ImageProcessorConfig, description=_DESCRIPTION)
class ImageProcessor(Preprocessor):
    """
    General image processor to perform sequential transforms on a list of images.
    """

    required_backends = _required_backends

    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER
    image_processor_config_file = DEFAULT_IMAGE_PROCESSOR_CONFIG_FILE

    def __init__(self, config: ImageProcessorConfig, **kwargs):
        """
        Initializes the ImageProcessor.

        Args:
            config (ImageProcessorConfig): Configuration for the image processor.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, **kwargs)

    def __call__(
        self,
        images: List,
        device: str = None,
        mean: float = None,
        std: float = None,
        rescale: float = None,
        size: Tuple[int, int] = None,
        resample: float = None,
        mirror: bool = None,
        gray_scale: bool = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """
        Perform sequential image processing on a list of input images.

        Args:
            images (List): A list of input images of types torch, numpy, pillow.
            mean (float): Image mean value for normalization.
            std (float): Image std value for normalization.
            rescale (float): Scale factor for rescaling the image.
            size (Tuple[int, int]): Image size tuple (width, height) for resizing.
            resample (float): Resample method value based on Image.Resampling.
            mirror (bool): Flag to mirror the images.
            gray_scale (bool): Flag to convert images to grayscale.
            return_tensors (str): The type of the output images.
            **kwargs: Extra parameters.

        Returns:
            dict: Transformed images list.
        """
        mean = mean or self.config.mean
        std = std or self.config.std
        rescale = rescale or self.config.rescale
        size = size or self.config.size
        resample = resample or self.config.resample
        mirror = mirror or self.config.mirror
        gray_scale = gray_scale or self.config.gray_scale

        if not isinstance(images, Iterable) or isinstance(images, str):
            images = [images]

        # Load images if inputs are list of files
        images = [load_image(x, return_type="numpy") if isinstance(x, str) else x for x in images]

        # Cast image types
        images = [convert_image_type(image, target_type="numpy") for image in images]

        # Convert to grayscale
        if gray_scale:
            images = [gray_scale_image(image, return_type="numpy") for image in images]

        # Mirror images if mirror is set
        if mirror:
            images = [mirror_image(image, return_type="numpy") for image in images]

        if size is not None:
            if not isinstance(size, Iterable) or len(size) > 2:
                raise ValueError(f"The parameter `size` must be a tuple/list of (width, height), got `{size}`")
            images = [resize_image(image, size=size, resample=resample) for image in images]

        if rescale is not None:
            images = [rescale_image(image, scale=rescale) for image in images]

        if mean is not None and std is not None:
            images = [normalize_image(image, mean=mean, std=std, channel_axis="last") for image in images]

        # Transpose channels axis
        images = [transpose_channels_axis_side(image, axis_side="first") for image in images]

        # Return images batch dict
        images = np.array([convert_image_type(image, target_type="numpy") for image in images], dtype=np.float32)

        images = convert_batch_dict_dtype({"pixel_values": images}, dtype=return_tensors)

        if device:
            import torch

            images = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in images.items()}

        return images

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder: str = None,
        config_filename: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> "ImageProcessor":
        """
        Load an ImageProcessor from a specified path.

        Args:
            hub_or_local_path: Path to the hub or local location.
            subfolder (str): Subfolder within the specified path.
            config_filename (str): Configuration filename.
            cache_dir: Path to cache directory
            **kwargs: Additional keyword arguments.

        Returns:
            ImageProcessor: Loaded image processor instance.
        """
        subfolder = subfolder or cls.preprocessor_subfolder
        config_filename = config_filename or cls.image_processor_config_file
        config = ImageProcessorConfig.load(
            hub_or_local_path,
            filename=config_filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        image_processor = build_preprocessor(config.name, config, **kwargs)
        return image_processor

    def save(
        self,
        path,
        subfolder=None,
        config_filename=None,
    ):
        """
        Save the ImageProcessor configuration.

        Args:
            path: Path to save the configuration.
            subfolder (str): Subfolder within the specified path.
            config_filename (str): Configuration filename.
        """
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
        """
        Push the ImageProcessor configuration to the hub.

        Args:
            repo_id: ID of the repository.
            subfolder (str): Subfolder within the repository.
            commit_message (str): Commit message.
            private (bool): Flag indicating whether the repository is private.
            config_filename (str): Configuration filename.
        """
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
