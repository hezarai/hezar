from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from .vit_config import ViTConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import ViTConfig as ViTConfig_
    from transformers import ViTModel

if is_backend_available(Backends.PILLOW):
    from PIL import Image

_required_backends = [Backends.TRANSFORMERS, Backends.TOKENIZERS, Backends.PILLOW]


@register_model("vit", config_class=ViTConfig)
class ViT(Model):
    required_backends = _required_backends
    image_processor = "image_processor"
    loss_func_name = "cross_entropy"

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.vit = ViTModel(ViTConfig_(**self.config))

    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
    ):
        outputs = self.vit(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return outputs

    def preprocess(self, inputs: List[str] | List[np.ndarray] | List["Image"] | List[torch.Tensor], **kwargs):
        image_processor = self.preprocessor[self.image_processor]
        processed_outputs = image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(self, model_outputs: Dict[str, torch.Tensor]):
        outputs = {
            "last_hidden_state": model_outputs.get("last_hidden_state", None),
            "pooler_output": model_outputs.get("pooler_output", None),
            "hidden_states": model_outputs.get("hidden_states", None),
            "attentions": model_outputs.get("attentions", None),
        }
        return outputs
