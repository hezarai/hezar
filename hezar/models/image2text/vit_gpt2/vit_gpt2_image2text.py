from __future__ import annotations

from typing import List

import numpy as np
import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import Image2TextOutput
from .vit_gpt2_image2text_config import ViTGPT2Image2TextConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import (
        GenerationConfig,
        GPT2Config,
        GPT2LMHeadModel,
        VisionEncoderDecoderModel,
        ViTConfig,
        ViTModel,
    )

if is_backend_available(Backends.PILLOW):
    from PIL import Image

_required_backends = [Backends.TRANSFORMERS, Backends.TOKENIZERS, Backends.PILLOW]


@register_model("vit_gpt2_image2text", config_class=ViTGPT2Image2TextConfig)
class ViTGPT2Image2Text(Model):
    """
    ViT + GPT2 for image to text generation (image captioning)
    """

    is_generative = True
    required_backends = _required_backends
    image_processor = "image_processor"
    tokenizer_name = "bpe_tokenizer"
    loss_func_name = "cross_entropy"

    def __init__(self, config: ViTGPT2Image2TextConfig, **kwargs):
        super().__init__(config, **kwargs)
        encoder = ViTModel(config=ViTConfig(**self.config.encoder))
        decoder = GPT2LMHeadModel(config=GPT2Config(**self.config.decoder))
        self.vit_gpt2 = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    def forward(
        self,
        pixel_values,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        outputs = self.vit_gpt2(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(logits.reshape(-1, self.vit_gpt2.decoder.config.vocab_size), labels.reshape(-1))
        return loss

    def generate(self, pixel_values, generation_config=None, **kwargs):
        if generation_config is None:
            generation_config = self.config.dict()["generation"]
        generation_config = GenerationConfig(**generation_config)
        outputs = self.vit_gpt2.generate(inputs=pixel_values, generation_config=generation_config, **kwargs)

        return outputs

    def preprocess(self, inputs: List[str] | List[np.ndarray] | List["Image"] | List[torch.Tensor], **kwargs):
        image_processor = self.preprocessor[self.image_processor]
        processed_outputs = image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(self, model_outputs, **kwargs):
        tokenizer = self.preprocessor[self.tokenizer_name]
        decoded_outputs = tokenizer.decode(model_outputs.cpu().numpy().tolist())
        outputs = [Image2TextOutput(text=text) for text in decoded_outputs]
        return outputs
