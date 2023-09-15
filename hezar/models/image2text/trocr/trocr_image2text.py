from typing import List, Union

import numpy as np
import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import GenerativeModel
from ...model_outputs import Image2TextOutput
from .trocr_image2text_config import TrOCRImage2TextConfig

if is_backend_available(Backends.TRANSFORMERS):
    from transformers import (
        GenerationConfig,
        RobertaConfig,
        RobertaForCausalLM,
        VisionEncoderDecoderModel,
        ViTConfig,
        ViTModel,
    )

if is_backend_available(Backends.PILLOW):
    from PIL import Image

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
    Backends.PILLOW
]


@register_model("trocr_image2text", config_class=TrOCRImage2TextConfig)
class TrOCRImage2Text(GenerativeModel):
    """
    TrOCR for optical character recognition
    """
    required_backends = _required_backends
    image_processor = "image_processor"
    tokenizer = "bpe_tokenizer"

    def __init__(self, config: TrOCRImage2TextConfig, **kwargs):
        super().__init__(config, **kwargs)
        encoder = ViTModel(config=ViTConfig(**self.config.encoder))
        decoder = RobertaForCausalLM(config=RobertaConfig(**self.config.decoder))
        self.trocr = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    def forward(self, inputs, **kwargs):
        pixel_values = inputs.get("pixel_values", None)
        decoder_input_ids = inputs.get("decoder_input_ids", None)
        decoder_attention_mask = inputs.get("decoder_attention_mask", None)
        encoder_outputs = inputs.get("encoder_outputs", None)
        past_key_values = inputs.get("past_key_values", None)
        decoder_inputs_embeds = inputs.get("decoder_inputs_embeds", None)
        labels = inputs.get("labels", None)
        use_cache = inputs.get("use_cache", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        outputs = self.trocr(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    def generate(self, inputs, **kwargs):
        input_ids = inputs.get("pixel_values", None)
        generation_config = inputs.get("generation_config", None)
        if generation_config is None:
            generation_config = self.config.dict()["generation"]
        generation_config = GenerationConfig(**generation_config)
        outputs = self.trocr.generate(inputs=input_ids, generation_config=generation_config, **kwargs)

        return outputs

    def preprocess(self, inputs: Union[List[str], List[np.ndarray], List["Image"], List[torch.Tensor]], **kwargs):
        image_processor = self.preprocessor[self.image_processor]
        processed_outputs = image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(self, inputs, **kwargs):
        tokenizer = self.preprocessor[self.tokenizer]
        decoded_outputs = tokenizer.decode(inputs.numpy().tolist())
        return Image2TextOutput(texts=decoded_outputs)
