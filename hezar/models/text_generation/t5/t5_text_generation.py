from __future__ import annotations

from typing import Dict, List

import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import TextGenerationOutput
from .t5_text_generation_config import T5TextGenerationConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import T5Config, T5ForConditionalGeneration

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("t5_text_generation", config_class=T5TextGenerationConfig)
class T5TextGeneration(Model):
    """
    T5 for text to text generation
    """

    is_generative = True
    required_backends = _required_backends
    tokenizer_name = "sentencepiece_unigram_tokenizer"
    loss_func_name = "cross_entropy"

    def __init__(self, config: T5TextGenerationConfig, **kwargs):
        super().__init__(config=config, **kwargs)

        self.t5 = T5ForConditionalGeneration(T5Config(**self.config))

    def forward(
        self,
        token_ids,
        labels=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ) -> Dict:

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        outputs = self.t5(
            input_ids=token_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return dict(outputs)

    def _shift_right(self, input_ids):
        return self.t5._shift_right(input_ids)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.clone()
        labels[labels == self.config.pad_token_id] = -100
        loss = self.loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def generate(self, token_ids, attention_mask=None, **kwargs):
        # TODO Merge kwargs into generation config so users can control generation from kwargs
        model_inputs = {"input_ids": token_ids, "attention_mask": attention_mask}
        generation_kwargs = {"min_length": self.config.min_length, "max_length": self.config.max_length}
        output_ids = self.t5.generate(**model_inputs, **generation_kwargs)
        return output_ids

    def preprocess(self, inputs: str | List[str], prefix=None):
        if isinstance(inputs, str):
            inputs = [inputs]
        prefix = prefix or self.config.input_prefix
        if prefix:
            inputs = [f"{prefix}{x}" for x in inputs]
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, generated_ids: torch.Tensor, **kwargs):
        tokenizer = self.preprocessor[self.tokenizer_name]
        decoded_outputs = tokenizer.decode(generated_ids.cpu().numpy().tolist())
        outputs = [TextGenerationOutput(text=text) for text in decoded_outputs]
        return outputs
