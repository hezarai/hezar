from dataclasses import dataclass

from ....configs import ModelConfig


@dataclass
class CraftTextDetectionConfig(ModelConfig):
    name = "craft_text_detection"
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    poly: bool = False
