from dataclasses import dataclass
from typing import Dict

from ....configs import ModelConfig


@dataclass
class CRNNImage2TextConfig(ModelConfig):
    name = "crnn_image2text"
    id2label: Dict[int, str] = None
    blank_id: int = 0
    n_channels: int = 1
    image_height: int = 32
    image_width: int = 128
    map2seq_in_dim: int = 2048
    map2seq_out_dim: int = 64
    rnn_dim: int = 256
    reverse_prediction_text: bool = None
    reverse_output_digits: bool = None
