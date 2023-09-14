from dataclasses import dataclass
from typing import Dict

from ....configs import ModelConfig


@dataclass
class CRNNImage2TextConfig(ModelConfig):
    name = "crnn_image2text"
    id2label: Dict[int, str] = None
    n_channels: int = 1
    img_height: int = 50
    map2seq_dim: int = 64
    rnn_dim: int = 256
    max_new_chars: int = 128

